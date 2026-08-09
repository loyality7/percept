package com.percept.app

import android.graphics.RectF
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlin.math.abs
import kotlin.math.hypot

// Portrait analysis frame — the sensor image is rotated to match before any CV runs.
private const val ANALYSIS_W = 360
private const val ANALYSIS_H = 480

private const val QUALITY_LEVEL = 0.01 // relative; the absolute gate is config.minEig

// Flow runs twice per frame (forward + backward check), so its cost is doubled and it
// dominates the budget. 15px/2 levels on a 360x480 frame still tracks a fast hand; 21px/3
// measured ~32ms/frame on-device, which capped CV at 18fps.
private val LK_WIN = Size(15.0, 15.0)
private const val LK_LEVELS = 2
private val LK_CRITERIA = TermCriteria(TermCriteria.EPS or TermCriteria.COUNT, 20, 0.03)

/**
 * Port of the validated Python pipeline (validator/tracker_validator.py). Every point,
 * line and box comes from the pixels of the current frame:
 *
 *  - MOG2 background subtraction isolates what is actually moving
 *  - Shi-Tomasi corners, seeded with a bias toward that foreground and gated by an
 *    absolute cornerMinEigenVal floor
 *  - Lucas-Kanade optical flow with a forward-backward consistency check
 *  - findContours on the same foreground for the boxes (not point clusters)
 *  - k-nearest linking under a distance cutoff, recomputed every frame
 *
 * A point that fails flow this frame is dropped, never frozen and never interpolated.
 * IDs are handed out once, at first detection, and travel with the point until it dies.
 *
 * Every large buffer is allocated once and reused. Allocating a Mat and a ByteArray per
 * frame is ~200KB of churn at 20fps, which shows up on-device as GC stutter rather than
 * as a slow frame — the kind of jank that is hard to attribute after the fact.
 */
class TrackerAnalyzer(
    private val config: TrackerConfig,
    private val onFrame: (TrackedFrame, Int, Int) -> Unit
) : ImageAnalysis.Analyzer {

    private var points = ArrayList<Point>()
    private var ids = ArrayList<Int>()
    private var speeds = ArrayList<Float>()
    private var bgFrames = ArrayList<Int>()
    private var nextId = 0
    private var frameNo = 0
    private var eigMax = 0.0
    private var fgRatio = 0.0

    private var seededFg = 0
    private var seededAll = 0

    // --- reusable buffers: allocated on first use, resized in place afterwards ---
    private var bytes = ByteArray(0)
    private val matFull = Mat()
    private val matRot = Mat()
    private val grayA = Mat()
    private val grayB = Mat()
    private var useA = true
    private var hasPrev = false
    private val fgMat = Mat()
    private val contourWork = Mat()
    private val hierarchy = Mat()
    private val gray32 = Mat()
    private val eig = Mat()
    private val maskMat = Mat()

    private val subtractor = Video.createBackgroundSubtractorMOG2(200, 32.0, false)
    private val openKernel =
        Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
    private val dilateKernel =
        Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(15.0, 15.0))
    private val closeKernel = Imgproc.getStructuringElement(
        Imgproc.MORPH_ELLIPSE,
        Size(config.closeKernel.toDouble(), config.closeKernel.toDouble())
    )

    override fun analyze(image: ImageProxy) {
        // The sensor delivers landscape frames on a portrait-held phone; rotate before
        // detection or every point lands on the wrong axis.
        val rotation = image.imageInfo.rotationDegrees
        val swapped = rotation == 90 || rotation == 270
        val outW = if (swapped) image.height else image.width
        val outH = if (swapped) image.width else image.height

        // Ping-pong the two gray buffers: the previous frame must stay intact for optical
        // flow while the current one is being written.
        val gray = if (useA) grayA else grayB
        val prev = if (useA) grayB else grayA
        useA = !useA

        val t0 = System.nanoTime()
        try {
            yPlaneToGray(image, rotation, gray)
        } finally {
            image.close()
        }
        val t1 = System.nanoTime()

        foreground(gray)
        // Above fgMaxRatio the camera itself moved, so "foreground" is the whole frame
        // and biasing toward it would be meaningless.
        val fgUsable = fgRatio > 0.0005 && fgRatio < config.fgMaxRatio
        val t2 = System.nanoTime()

        if (hasPrev) track(prev, gray)
        if (fgUsable && config.pruneBg == 1) pruneBackgroundPoints(fgMat)
        val t3 = System.nanoTime()

        if (points.size < config.minPoints || frameNo % config.redetectEvery == 0) {
            detect(gray, if (fgUsable) fgMat else null)
        }
        val t4 = System.nanoTime()

        val boxes = motionBoxes()
        val boxEdges = linkBoxes(boxes)
        val t5 = System.nanoTime()

        // Short edges describe one surface; long edges join two objects. Kept in one
        // array because both are point-to-point links drawn as the same web.
        val edges = linkNearby() + linkAcrossObjects(boxes, boxEdges)
        val t6 = System.nanoTime()

        val timings = Timings(
            convert = (t1 - t0) / 1e6f,
            flow = (t3 - t2) / 1e6f,
            detect = (t4 - t3) / 1e6f,
            boxes = (t5 - t4) / 1e6f + (t2 - t1) / 1e6f, // background subtraction + contours
            link = (t6 - t5) / 1e6f
        )

        hasPrev = true
        frameNo++

        // adb logcat -s Percept — the same numbers as the HUD, but recorded over time so
        // a pattern that flashes past on screen can be read back afterwards.
        if (frameNo % 30 == 0) {
            android.util.Log.d(
                "Percept",
                "f=$frameNo pts=${points.size} edges=${edges.size / 2} " +
                    "objects=${boxes.size} objLinks=${boxEdges.size / 2} " +
                    "fg=${"%.3f".format(fgRatio)} usable=$fgUsable eig=${"%.4f".format(eigMax)} " +
                    "seedFg=$seededFg seedAll=$seededAll | " +
                    "convert=${"%.1f".format(timings.convert)}ms flow=${"%.1f".format(timings.flow)}ms " +
                    "detect=${"%.1f".format(timings.detect)}ms boxes=${"%.1f".format(timings.boxes)}ms " +
                    "link=${"%.1f".format(timings.link)}ms total=${"%.1f".format(timings.total)}ms"
            )
        }

        onFrame(buildFrame(edges, boxes, boxEdges, outW, outH, timings), outW, outH)
    }

    /**
     * CameraX ImageAnalysis -> rotated, downscaled 8-bit grayscale Mat.
     *
     * Only the Y plane is read. Y *is* luminance, so this is a real grayscale image with
     * no conversion cost — every downstream operation (corners, flow, MOG2) wants
     * grayscale anyway, so converting YUV to BGR first would mean paying for colour
     * that then gets thrown away, plus the chroma-plane stride handling that is the
     * usual source of smearing artefacts.
     *
     * rowStride is not the same as width: rows are padded, so the buffer is read at its
     * true stride and the real pixels taken as a sub-matrix header (no copy).
     */
    private fun yPlaneToGray(image: ImageProxy, rotationDegrees: Int, dst: Mat) {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val n = buffer.remaining()

        if (bytes.size < n) bytes = ByteArray(n)
        buffer.get(bytes, 0, n)

        val rows = minOf(image.height, n / rowStride)
        matFull.create(rows, rowStride, CvType.CV_8UC1) // no-op when the shape is unchanged
        matFull.put(0, 0, bytes)

        val roi = matFull.submat(0, rows, 0, minOf(image.width, rowStride))
        when (rotationDegrees) {
            90 -> Core.rotate(roi, matRot, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(roi, matRot, Core.ROTATE_180)
            270 -> Core.rotate(roi, matRot, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> roi.copyTo(matRot)
        }
        roi.release() // header only; the pixel data belongs to matFull

        Imgproc.resize(matRot, dst, Size(ANALYSIS_W.toDouble(), ANALYSIS_H.toDouble()))
    }

    /**
     * MOG2 keeps a per-pixel model of the static scene; whatever stops matching it is the
     * subject. Opening removes speckle, dilation closes the gaps so a hand comes back as
     * one region instead of a scatter of finger fragments.
     */
    private fun foreground(gray: Mat) {
        subtractor.apply(gray, fgMat, config.learningRate)
        Imgproc.threshold(fgMat, fgMat, 200.0, 255.0, Imgproc.THRESH_BINARY) // drop shadows
        Imgproc.morphologyEx(fgMat, fgMat, Imgproc.MORPH_OPEN, openKernel)
        // CLOSE before DILATE: a subject with flat interior regions (a palm, a plain
        // sleeve) only differs from the background at its edges, so MOG2 returns a hollow
        // outline. Closing fills those interiors, which is what turns a ring of fragments
        // back into one solid blob the detector can seed inside.
        Imgproc.morphologyEx(fgMat, fgMat, Imgproc.MORPH_CLOSE, closeKernel)
        Imgproc.dilate(fgMat, fgMat, dilateKernel)
        fgRatio = Core.countNonZero(fgMat).toDouble() / (fgMat.rows() * fgMat.cols())
    }

    /**
     * Lucas-Kanade, forward then backward. A point survives only if forward status is 1,
     * backward status is 1, it tracks back to within config.fbThreshold px of where it
     * started, and it is still inside the frame. The backward pass is what stops a point
     * sliding off a moving object onto the background.
     */
    private fun track(prev: Mat, gray: Mat) {
        if (points.isEmpty()) return

        val p0 = MatOfPoint2f().apply { fromList(points) }
        val p1 = MatOfPoint2f()
        val stFwd = MatOfByte()
        val errFwd = MatOfFloat()
        Video.calcOpticalFlowPyrLK(prev, gray, p0, p1, stFwd, errFwd, LK_WIN, LK_LEVELS, LK_CRITERIA)

        val pBack = MatOfPoint2f()
        val stBwd = MatOfByte()
        val errBwd = MatOfFloat()
        Video.calcOpticalFlowPyrLK(gray, prev, p1, pBack, stBwd, errBwd, LK_WIN, LK_LEVELS, LK_CRITERIA)

        val start = points
        val forward = p1.toArray()
        val back = pBack.toArray()
        val okFwd = stFwd.toArray()
        val okBwd = stBwd.toArray()

        val keptPts = ArrayList<Point>(forward.size)
        val keptIds = ArrayList<Int>(forward.size)
        val keptSpeeds = ArrayList<Float>(forward.size)
        val keptBg = ArrayList<Int>(forward.size)
        for (i in forward.indices) {
            if (okFwd[i].toInt() != 1 || okBwd[i].toInt() != 1) continue
            val fbErr = maxOf(abs(start[i].x - back[i].x), abs(start[i].y - back[i].y))
            if (fbErr >= config.fbThreshold) continue
            val p = forward[i]
            if (p.x < 0 || p.y < 0 || p.x >= ANALYSIS_W || p.y >= ANALYSIS_H) continue
            keptPts.add(p)
            keptIds.add(ids[i])
            keptSpeeds.add(hypot(p.x - start[i].x, p.y - start[i].y).toFloat())
            keptBg.add(bgFrames[i])
        }

        p0.release(); p1.release(); stFwd.release(); errFwd.release()
        pBack.release(); stBwd.release(); errBwd.release()

        points = keptPts
        ids = keptIds
        speeds = keptSpeeds
        bgFrames = keptBg
    }

    /**
     * Retire points that have sat on static background for bgGrace consecutive frames.
     * Off by default (it thins the web); the grace period is what lets a subject pause
     * mid-gesture without being dropped.
     */
    private fun pruneBackgroundPoints(fg: Mat) {
        var w = 0
        for (i in points.indices) {
            val x = points[i].x.toInt().coerceIn(0, fg.cols() - 1)
            val y = points[i].y.toInt().coerceIn(0, fg.rows() - 1)
            val onFg = fg.get(y, x)[0] > 0
            val age = if (onFg) 0 else bgFrames[i] + 1
            if (age > config.bgGrace) continue

            points[w] = points[i]
            ids[w] = ids[i]
            speeds[w] = speeds[i]
            bgFrames[w] = age
            w++
        }
        while (points.size > w) {
            val last = points.size - 1
            points.removeAt(last); ids.removeAt(last)
            speeds.removeAt(last); bgFrames.removeAt(last)
        }
    }

    /**
     * Shi-Tomasi corners, gated by the absolute cornerMinEigenVal floor, in two passes.
     *
     * Shi-Tomasi picks the most textured thing in frame and furniture beats skin every
     * time, so an unbiased detector spends its whole budget on the room. Pass one
     * reserves a share for the moving subject; pass two fills the remainder from the
     * whole frame so the web still covers the scene.
     */
    private fun detect(gray: Mat, fg: Mat?) {
        if (config.maxPoints - points.size <= 0) return

        // Normalise to 0..1 so minEig is dimensionless and survives exposure changes.
        gray.convertTo(gray32, CvType.CV_32F, 1.0 / 255.0)
        Imgproc.cornerMinEigenVal(gray32, eig, 3, 3)
        eigMax = Core.minMaxLoc(eig).maxVal

        val before = points.size
        if (fg != null && config.fgBias > 0.0) {
            val quota = ((config.maxPoints - points.size) * config.fgBias).toInt()
            if (quota > 0) seed(gray, region = fg, count = quota)
        }
        seededFg = points.size - before

        val afterFg = points.size
        val rest = config.maxPoints - points.size
        if (rest > 0) seed(gray, region = null, count = rest)
        seededAll = points.size - afterFg
    }

    /** One Shi-Tomasi pass inside `region` (null = whole frame), gated by the eig floor. */
    private fun seed(gray: Mat, region: Mat?, count: Int) {
        if (region != null) {
            region.copyTo(maskMat)
        } else {
            maskMat.create(gray.size(), CvType.CV_8UC1)
            maskMat.setTo(Scalar(255.0))
        }
        // Mask out ground already held by live points so re-detection doesn't stack
        // duplicate IDs on the same texture.
        for (p in points) {
            Imgproc.circle(maskMat, p, config.minDistance.toInt(), Scalar(0.0), -1)
        }

        val corners = MatOfPoint()
        Imgproc.goodFeaturesToTrack(
            gray, corners, count, QUALITY_LEVEL, config.minDistance, maskMat, 7
        )

        for (p in corners.toArray()) {
            val x = p.x.toInt().coerceIn(0, eig.cols() - 1)
            val y = p.y.toInt().coerceIn(0, eig.rows() - 1)
            if (eig.get(y, x)[0] < config.minEig) continue
            points.add(p)
            ids.add(nextId++)
            speeds.add(0f)
            bgFrames.add(0)
        }
        corners.release()
    }

    /**
     * Boxes come from the real foreground blobs, not from clusters of tracked points.
     * Hold still and the boxes go away, because nothing is moving.
     */
    private fun motionBoxes(): List<RectF> {
        fgMat.copyTo(contourWork) // findContours mutates its input
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(
            contourWork, contours, hierarchy,
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
        )

        val boxes = ArrayList<RectF>()
        for (c in contours) {
            if (Imgproc.contourArea(c) >= config.minBoxArea) {
                val r = Imgproc.boundingRect(c)
                boxes.add(
                    RectF(
                        r.x.toFloat(), r.y.toFloat(),
                        (r.x + r.width).toFloat(), (r.y + r.height).toFloat()
                    )
                )
            }
            c.release()
        }
        return mergeBoxes(boxes)
    }

    /**
     * Link detected objects to each other: every box to its nearest `objLinks` boxes,
     * measured centre to centre. Two people moving through frame get a line drawn between
     * them — the relationship the whole look is built on, and one that point-level edges
     * can never produce, since those only ever connect texture within a single object.
     */
    private fun linkBoxes(boxes: List<RectF>): IntArray {
        val n = boxes.size
        if (n < 2) return IntArray(0)

        val cx = FloatArray(n) { boxes[it].centerX() }
        val cy = FloatArray(n) { boxes[it].centerY() }
        val out = ArrayList<Int>()
        val seen = HashSet<Long>()
        val k = minOf(config.objLinks, n - 1)

        for (i in 0 until n) {
            repeat(k) {
                var best = -1
                var bestD = Float.MAX_VALUE
                for (j in 0 until n) {
                    if (i == j) continue
                    val key = if (i < j) i.toLong() * n + j else j.toLong() * n + i
                    if (key in seen) continue
                    val dx = cx[i] - cx[j]
                    val dy = cy[i] - cy[j]
                    val d = dx * dx + dy * dy
                    if (d < bestD) { bestD = d; best = j }
                }
                if (best >= 0) {
                    val key = if (i < best) i.toLong() * n + best else best.toLong() * n + i
                    seen.add(key)
                    out.add(i); out.add(best)
                }
            }
        }
        return out.toIntArray()
    }

    /**
     * Fuse boxes that touch or nearly touch. A single object rarely comes back as one
     * contour — a hand splits at the fingers, a torso splits at a fold — and drawing each
     * fragment separately reads as "the detector found seven random things" rather than
     * one object. Repeats until stable, since merging two boxes can bring a third within
     * range.
     */
    private fun mergeBoxes(input: List<RectF>): List<RectF> {
        val boxes = ArrayList(input)
        val m = config.boxMerge.toFloat()
        var merged = true
        while (merged) {
            merged = false
            outer@ for (i in boxes.indices) {
                for (j in i + 1 until boxes.size) {
                    val a = boxes[i]
                    val b = boxes[j]
                    val near = a.left - m < b.right && b.left - m < a.right &&
                        a.top - m < b.bottom && b.top - m < a.bottom
                    if (!near) continue
                    boxes[i] = RectF(
                        minOf(a.left, b.left), minOf(a.top, b.top),
                        maxOf(a.right, b.right), maxOf(a.bottom, b.bottom)
                    )
                    boxes.removeAt(j)
                    merged = true
                    break@outer
                }
            }
        }
        return boxes
    }

    /**
     * Web edges as index pairs into `points`, recomputed from scratch every frame.
     * Each point keeps only its `maxLinks` nearest neighbours inside `maxEdge`; a plain
     * radius cutoff turns dense texture into a solid blanket of lines.
     *
     * Indices rather than coordinates: the overlay draws faster than CV runs, so it
     * rebuilds the lines from the interpolated point positions it is actually showing.
     * ponytail: O(n^2) over <=300 points is ~45k checks; spatial hash if the cap rises.
     */
    private fun linkNearby(): IntArray {
        val n = points.size
        if (n < 2) return IntArray(0)

        val out = ArrayList<Int>()
        val seen = HashSet<Long>()

        // Squared distances throughout: hypot() per pair was a measurable slice of the
        // frame budget, and ordering by d^2 is the same ordering as by d.
        val xs = DoubleArray(n) { points[it].x }
        val ys = DoubleArray(n) { points[it].y }
        val localMax2 = config.maxEdge * config.maxEdge

        // Top-k by insertion rather than sorting every candidate: only maxLinks of them
        // survive, so sorting the rest was wasted work (and boxed every distance).
        val k = config.maxLinks
        val bestD = DoubleArray(k)
        val bestJ = IntArray(k)

        for (i in 0 until n) {
            var filled = 0
            val ax = xs[i]
            val ay = ys[i]
            for (j in 0 until n) {
                if (i == j) continue
                val dx = ax - xs[j]
                val dy = ay - ys[j]
                val d2 = dx * dx + dy * dy
                if (d2 > localMax2) continue

                if (filled < k) {
                    var p = filled++
                    while (p > 0 && bestD[p - 1] > d2) {
                        bestD[p] = bestD[p - 1]; bestJ[p] = bestJ[p - 1]; p--
                    }
                    bestD[p] = d2; bestJ[p] = j
                } else if (d2 < bestD[k - 1]) {
                    var p = k - 1
                    while (p > 0 && bestD[p - 1] > d2) {
                        bestD[p] = bestD[p - 1]; bestJ[p] = bestJ[p - 1]; p--
                    }
                    bestD[p] = d2; bestJ[p] = j
                }
            }
            for (m in 0 until filled) {
                val j = bestJ[m]
                val key = if (i < j) i.toLong() * n + j else j.toLong() * n + i
                if (seen.add(key)) { out.add(i); out.add(j) }
            }
        }

        return out.toIntArray()
    }

    /**
     * Long edges, and the rule for where they belong: a long line is only drawn *between
     * two different detected objects*, never inside one and never between arbitrary
     * points.
     *
     * Short and long are answering different questions. Short k-nearest edges describe
     * texture — the structure of one surface. A long edge asserts a relationship between
     * two separate things, so it has to have two separate things to join. Picking long
     * pairs by distance alone (the previous approach) produced spans that crossed the
     * frame for no reason and read as noise, because length was the only criterion.
     *
     * Endpoints are the real tracked point in each object closest to the other object,
     * so the line spans the actual gap between them rather than starting somewhere
     * arbitrary inside each blob.
     */
    private fun linkAcrossObjects(boxes: List<RectF>, boxEdges: IntArray): IntArray {
        if (boxes.size < 2 || boxEdges.isEmpty() || points.isEmpty()) return IntArray(0)

        val members = Array(boxes.size) { ArrayList<Int>() }
        for (i in points.indices) {
            val px = points[i].x.toFloat()
            val py = points[i].y.toFloat()
            for (b in boxes.indices) {
                if (boxes[b].contains(px, py)) { members[b].add(i); break }
            }
        }

        val out = ArrayList<Int>()
        var e = 0
        while (e < boxEdges.size && out.size / 2 < config.longLinks) {
            val a = boxEdges[e]
            val b = boxEdges[e + 1]
            val ma = members[a]
            val mb = members[b]
            if (ma.isNotEmpty() && mb.isNotEmpty()) {
                val bc = boxes[b]
                val ac = boxes[a]
                val pa = ma.minByOrNull { hypot(points[it].x - bc.centerX(), points[it].y - bc.centerY()) }!!
                val pb = mb.minByOrNull { hypot(points[it].x - ac.centerX(), points[it].y - ac.centerY()) }!!
                val d = hypot(points[pa].x - points[pb].x, points[pa].y - points[pb].y)
                if (d >= config.longMin) { out.add(pa); out.add(pb) }
            }
            e += 2
        }
        return out.toIntArray()
    }

    /** Scale everything from the analysis frame up to the rotated camera frame. */
    private fun buildFrame(
        edges: IntArray,
        boxes: List<RectF>,
        boxEdges: IntArray,
        outW: Int,
        outH: Int,
        timings: Timings
    ): TrackedFrame {
        val sx = outW.toFloat() / ANALYSIS_W
        val sy = outH.toFloat() / ANALYSIS_H

        val tracked = points.mapIndexed { i, p ->
            TrackedPoint(
                ids[i],
                (p.x * sx).toFloat(),
                (p.y * sy).toFloat(),
                speeds.getOrElse(i) { 0f }
            )
        }
        val scaledBoxes = boxes.map {
            RectF(it.left * sx, it.top * sy, it.right * sx, it.bottom * sy)
        }
        return TrackedFrame(tracked, edges, scaledBoxes, boxEdges, eigMax, fgRatio, timings)
    }
}
