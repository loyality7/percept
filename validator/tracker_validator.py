#!/usr/bin/env python3
"""
Real-time OpenCV feature tracking validator.

Every dot, line, box and number drawn on screen is computed from the pixels of
the current frame. Nothing is invented, randomised, smoothed-in, or held over
from a previous frame as a guess. Structure mirrors the Android port 1:1:

    FrameSource        -> CameraX ImageAnalysis
    FeatureDetector    -> Imgproc.goodFeaturesToTrack + Imgproc.cornerMinEigenVal
    OpticalFlowTracker -> Video.calcOpticalFlowPyrLK
    MotionBoxDetector  -> Video.createBackgroundSubtractorMOG2 + Imgproc.findContours
    GeometryLinker     -> Delaunay (Kotlin: Subdiv2D) or nearest-neighbour loop
    Renderer           -> Canvas overlay View

    pip install opencv-python numpy scipy
"""

import argparse
import time

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError

# Lucas-Kanade parameters, shared by the forward and backward pass.
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class FrameSource:
    """Grabs a real webcam frame, downscales it, and produces the grayscale
    float image every downstream CV operation reads from. No synthesis here —
    if the camera gives nothing, read() returns None and the loop ends."""

    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width, self.height = width, height

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None, None
        if (bgr.shape[1], bgr.shape[0]) != (self.width, self.height):
            bgr = cv2.resize(bgr, (self.width, self.height))
        # 8-bit: calcOpticalFlowPyrLK and MOG2 both require CV_8U.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return bgr, gray

    def release(self):
        self.cap.release()


class FeatureDetector:
    """Shi-Tomasi corner detection on the actual current frame.

    goodFeaturesToTrack alone is NOT enough to prove a point is real: its
    qualityLevel is relative to the strongest corner *in that frame*, so a
    blank wall or a dark room promotes sensor noise to "best corner" and you
    get a full mesh out of nothing. So every candidate is re-checked against
    cornerMinEigenVal — the absolute Shi-Tomasi response at that pixel — and
    dropped if the local texture is below `min_eig`. Flat surface => no points."""

    def __init__(self, max_corners=300, quality=0.01, min_distance=10, min_eig=1e-3):
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance
        self.min_eig = min_eig
        self.last_eig_max = 0.0

    def detect(self, gray, mask=None, max_corners=None):
        # /255 makes the eigenvalues dimensionless, so one absolute texture
        # threshold holds across exposure and brightness changes.
        eig = cv2.cornerMinEigenVal(gray.astype(np.float32) / 255.0, blockSize=3, ksize=3)
        self.last_eig_max = float(eig.max())

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners or self.max_corners,
            qualityLevel=self.quality,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=7,
        )
        if pts is None:
            return np.empty((0, 2), np.float32)

        pts = pts.reshape(-1, 2).astype(np.float32)
        xs = np.clip(pts[:, 0].round().astype(int), 0, eig.shape[1] - 1)
        ys = np.clip(pts[:, 1].round().astype(int), 0, eig.shape[0] - 1)
        return pts[eig[ys, xs] >= self.min_eig]


class OpticalFlowTracker:
    """Lucas-Kanade optical flow between the previous real frame and the current
    real frame. A point survives only if, this frame:

      * forward flow reported status == 1, and
      * tracking it back to the previous frame lands within `fb_threshold` px
        of where it started (forward-backward check — this is what stops a
        point sliding off a moving hand onto the background), and
      * it is still inside the image.

    Anything else is dropped outright. Nothing is frozen at its last known
    position and nothing is interpolated. IDs are assigned once, at first
    detection, and travel with the point until it dies."""

    def __init__(self, fb_threshold=1.0):
        self.points = np.empty((0, 2), np.float32)
        self.ids = np.empty((0,), np.int64)
        self._next_id = 0

    def track(self, prev_gray, gray, fb_threshold=1.0):
        if len(self.points) == 0:
            return

        p0 = self.points.reshape(-1, 1, 2)
        p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
        p0_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK_PARAMS)

        fb_err = np.abs(p0 - p0_back).reshape(-1, 2).max(axis=1)
        h, w = gray.shape
        xy = p1.reshape(-1, 2)

        good = (
            (st_fwd.reshape(-1) == 1)
            & (st_bwd.reshape(-1) == 1)
            & (fb_err < fb_threshold)
            & (xy[:, 0] >= 0) & (xy[:, 0] < w)
            & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        )

        self.points = xy[good]
        self.ids = self.ids[good]

    def add(self, new_points):
        """Register freshly detected corners with new persistent IDs."""
        if len(new_points) == 0:
            return
        new_ids = np.arange(self._next_id, self._next_id + len(new_points), dtype=np.int64)
        self._next_id += len(new_points)
        self.points = np.vstack([self.points, new_points]).astype(np.float32)
        self.ids = np.concatenate([self.ids, new_ids])

    def occupancy_mask(self, shape, radius):
        """255 everywhere a new corner is allowed — i.e. not already tracked.
        Keeps re-detection from stacking duplicate IDs on the same texture."""
        mask = np.full(shape, 255, np.uint8)
        for x, y in self.points:
            cv2.circle(mask, (int(x), int(y)), radius, 0, -1)
        return mask


class MotionBoxDetector:
    """Boxes come from real moving pixels, not from clusters of tracked points.

    MOG2 background subtraction builds a per-pixel model of the static scene;
    pixels that stop matching it become the foreground mask. Morphological
    opening removes single-pixel noise, findContours traces the actual blob
    outlines, and boundingRect wraps each contour above `min_area`. Stand
    still => no boxes."""

    def __init__(self, min_area=800):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=32, detectShadows=False
        )
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, gray):
        fg = self.subtractor.apply(gray)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= self.min_area
        ]


class GeometryLinker:
    """Lines are recomputed from scratch every frame from the current point
    positions. Delaunay triangulation gives the natural web; edges longer than
    `max_edge` px are cut because a triangulation of a sparse set otherwise
    spans the whole screen. Nothing is cached between frames."""

    def __init__(self, mode="delaunay", max_edge=80.0):
        self.mode = mode
        self.max_edge = max_edge

    def link(self, points):
        n = len(points)
        if n < 2:
            return []
        pairs = self._delaunay_pairs(points) if self.mode == "delaunay" and n >= 4 \
            else self._neighbour_pairs(points)

        out = []
        for i, j in pairs:
            if np.hypot(*(points[i] - points[j])) <= self.max_edge:
                out.append((i, j))
        return out

    def _delaunay_pairs(self, points):
        try:
            tri = Delaunay(points)
        except QhullError:  # all points collinear/degenerate this frame
            return self._neighbour_pairs(points)
        pairs = set()
        for a, b, c in tri.simplices:
            pairs.update({tuple(sorted(e)) for e in ((a, b), (b, c), (c, a))})
        return pairs

    def _neighbour_pairs(self, points):
        # O(n^2) over a few hundred points — same loop the Kotlin port will use.
        # ponytail: fine at MAX_POINTS<=400; spatial hash if the cap ever rises.
        d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        i, j = np.where(np.triu(d, 1) > 0)
        return list(zip(i.tolist(), j.tolist()))


class Renderer:
    """One colour per layer so each can be verified independently:
    points white, lines gray, boxes red, IDs cyan."""

    def draw(self, bgr, points, ids, edges, boxes, hud):
        for i, j in edges:
            cv2.line(bgr, tuple(points[i].astype(int)), tuple(points[j].astype(int)),
                     (140, 140, 140), 1, cv2.LINE_AA)
        for x, y, w, h in boxes:
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        for (x, y), pid in zip(points, ids):
            cv2.circle(bgr, (int(x), int(y)), 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(bgr, str(pid), (int(x) + 4, int(y) - 4),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(bgr, hud, (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (0, 255, 0), 1, cv2.LINE_AA)
        return bgr


def run(args):
    source = FrameSource(args.camera, args.width, args.height)
    detector = FeatureDetector(args.max_points, args.quality, args.min_distance, args.min_eig)
    tracker = OpticalFlowTracker()
    boxer = MotionBoxDetector(args.min_area)
    linker = GeometryLinker(args.link_mode, args.max_edge)
    renderer = Renderer()

    prev_gray = None
    frame_no = 0
    t_last = time.time()
    fps = 0.0

    while True:
        bgr, gray = source.read()
        if gray is None:
            break

        if prev_gray is not None:
            tracker.track(prev_gray, gray, args.fb_threshold)

        # Re-seed on a fixed interval or whenever real tracking has thinned out.
        if len(tracker.points) < args.min_points or frame_no % args.redetect_every == 0:
            room = args.max_points - len(tracker.points)
            if room > 0:
                mask = tracker.occupancy_mask(gray.shape, args.min_distance)
                tracker.add(detector.detect(gray, mask=mask, max_corners=room))

        boxes = boxer.detect(gray)
        edges = linker.link(tracker.points)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
        t_last = now

        hud = (f"pts {len(tracker.points):3d}  edges {len(edges):4d}  boxes {len(boxes):2d}  "
               f"eig_max {detector.last_eig_max:.4f}  thr {args.min_eig:.4f}  {fps:4.1f}fps")
        cv2.imshow("tracker validator (q to quit)",
                   renderer.draw(bgr, tracker.points, tracker.ids, edges, boxes, hud))

        if frame_no % 10 == 0:
            print(hud, flush=True)

        prev_gray = gray
        frame_no += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    source.release()
    cv2.destroyAllWindows()


def selftest():
    """Asserts the three claims that matter, on synthetic frames with known
    ground truth. Run: python tracker_validator.py --selftest"""
    rng = np.random.default_rng(0)

    # 1. Flat frame => no points. Even with sensor-grade noise added.
    flat = np.clip(128 + rng.normal(0, 2, (240, 320)), 0, 255).astype(np.uint8)
    det = FeatureDetector(min_eig=1e-3)
    assert len(det.detect(flat)) == 0, f"blank frame produced {len(det.detect(flat))} points"

    # 2. Textured frame => points, and they land on the texture (right half only).
    tex = np.full((240, 320), 128, np.uint8)
    tex[:, 160:] = rng.integers(0, 256, (240, 160), dtype=np.uint8)
    pts = det.detect(tex)
    assert len(pts) > 20, f"textured frame produced only {len(pts)} points"
    assert (pts[:, 0] > 150).all(), "points landed on the flat half"

    # 3. Shift the texture by a known amount => optical flow reports that shift.
    dx = 4
    shifted = np.roll(tex, dx, axis=1)
    tr = OpticalFlowTracker()
    tr.add(pts[pts[:, 0] < 300])
    before = {int(i): float(x) for i, (x, _) in zip(tr.ids, tr.points)}
    tr.track(tex, shifted)
    assert len(tr.points) > 10, "flow dropped everything on a clean shift"
    measured = np.median([x - before[int(i)] for i, (x, _) in zip(tr.ids, tr.points)])
    assert abs(measured - dx) < 1.0, f"flow measured {measured:.2f}px, expected {dx}"

    # 4. Linker honours the distance cutoff and caches nothing.
    line = np.array([[0, 0], [10, 0], [200, 0], [210, 0]], np.float32)
    edges = GeometryLinker(max_edge=20).link(line)
    assert (0, 1) in edges and (2, 3) in edges and (1, 2) not in edges, edges

    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--max-points", type=int, default=300)
    p.add_argument("--min-points", type=int, default=50, help="re-detect below this count")
    p.add_argument("--redetect-every", type=int, default=25, help="frames between re-detects")
    p.add_argument("--quality", type=float, default=0.01, help="goodFeaturesToTrack qualityLevel")
    p.add_argument("--min-distance", type=int, default=10)
    p.add_argument("--min-eig", type=float, default=1e-3,
                   help="absolute texture floor; raise if a blank wall still shows points")
    p.add_argument("--fb-threshold", type=float, default=1.0,
                   help="forward-backward flow error, px")
    p.add_argument("--min-area", type=int, default=800, help="min contour area for a box")
    p.add_argument("--link-mode", choices=["delaunay", "neighbour"], default="delaunay")
    p.add_argument("--max-edge", type=float, default=80.0)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    selftest() if args.selftest else run(args)
