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

private val LK_WIN = Size(21.0, 21.0)
private const val LK_LEVELS = 3
private val LK_CRITERIA = TermCriteria(TermCriteria.EPS or TermCriteria.COUNT, 30, 0.01)

/**
 * Port of the validated Python pipeline (validator/tracker_validator.py). Every point,
 * line and box comes from the pixels of the current frame:
 *
 *  - MOG2 background subtraction isolates what is actually moving
 *  - Shi-Tomasi corners, seeded inside that foreground and gated by an absolute
 *    cornerMinEigenVal floor
 *  - Lucas-Kanade optical flow with a forward-backward consistency check
 *  - findContours on the same foreground for the boxes (not point clusters)
 *  - k-nearest linking under a distance cutoff, recomputed every frame
 *
 * A point that fails flow this frame is dropped, never frozen and never interpolated.
 * IDs are handed out once, at first detection, and travel with the point until it dies.
 */
class TrackerAnalyzer(
    private val config: TrackerConfig,
    private val onFrame: (TrackedFrame, Int, Int) -> Unit
) : ImageAnalysis.Analyzer {

    private var prevGray: Mat? = null
    private var points = ArrayList<Point>()
    private var ids = ArrayList<Int>()
    private var speeds = ArrayList<Float>()
    private var bgFrames = ArrayList<Int>()
    private var nextId = 0
    private var frameNo = 0
    private var eigMax = 0.0
    private var fgRatio = 0.0

    /** When true, each frame ships a bitmap of the raw MOG2 mask for the debug overlay. */
    @Volatile var wantMask = false
    private var maskBitmap: android.graphics.Bitmap? = null
    private var seededFg = 0
    private var seededAll = 0

    private val subtractor = Video.createBackgroundSubtractorMOG2(200, 32.0, false)
    private val openKernel =
        Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
    private val dilateKernel =
        Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(15.0, 15.0))

    override fun analyze(image: ImageProxy) {
        // The sensor delivers landscape frames on a portrait-held phone; rotate before
        // detection or every point lands on the wrong axis.
        val rotation = image.imageInfo.rotationDegrees
        val swapped = rotation == 90 || rotation == 270
        val outW = if (swapped) image.height else image.width
        val outH = if (swapped) image.width else image.height

        val gray = try {
            yPlaneToGray(image, rotation)
        } finally {
            image.close()
        }

        val fg = foreground(gray)
        // Above fgMaxRatio the camera itself moved, so "foreground" is the whole frame
        // and biasing toward it would be meaningless.
        val fgUsable = fgRatio > 0.0005 && fgRatio < config.fgMaxRatio

        val prev = prevGray
        if (prev != null) track(prev, gray)
        if (fgUsable && config.pruneBg == 1) pruneBackgroundPoints(fg)

        if (points.size < config.minPoints || frameNo % config.redetectEvery == 0) {
            detect(gray, if (fgUsable) fg else null)
        }

        val boxes = motionBoxes(fg)
        val lines = linkNearby()
        val mask = if (wantMask) maskToBitmap(fg) else null
        fg.release()

        prev?.release()
        prevGray = gray
        frameNo++

        // adb logcat -s Percept  — the same numbers as the HUD, but recorded over time so
        // a pattern that flashes past on screen can be read back afterwards.
        if (frameNo % 30 == 0) {
            android.util.Log.d(
                "Percept",
                "f=$frameNo pts=${points.size} lines=${lines.size / 4} boxes=${boxes.size} " +
                    "fg=${"%.3f".format(fgRatio)} usable=$fgUsable eigMax=${"%.4f".format(eigMax)} " +
                    "seedFg=$seededFg seedAll=$seededAll"
            )
        }

        onFrame(buildFrame(lines, boxes, outW, outH, mask), outW, outH)
    }

    /** CameraX ImageAnalysis -> rotated, downscaled 8-bit grayscale Mat. */
    private fun yPlaneToGray(image: ImageProxy, rotationDegrees: Int): Mat {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val bytes = ByteArray(buffer.remaining()).also { buffer.get(it) }

        val full = Mat(image.height, plane.rowStride, CvType.CV_8UC1)
        full.put(0, 0, bytes)
        val cropped = full.submat(0, image.height, 0, image.width).clone()
        full.release()

        val rotated = Mat()
        when (rotationDegrees) {
            90 -> Core.rotate(cropped, rotated, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(cropped, rotated, Core.ROTATE_180)
            270 -> Core.rotate(cropped, rotated, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> cropped.copyTo(rotated)
        }
        cropped.release()

        val resized = Mat()
        Imgproc.resize(rotated, resized, Size(ANALYSIS_W.toDouble(), ANALYSIS_H.toDouble()))
        rotated.release()
        return resized
    }

    /**
     * MOG2 keeps a per-pixel model of the static scene; whatever stops matching it is the
     * subject. Opening removes speckle, dilation closes the gaps so a hand comes back as
     * one region instead of a scatter of finger fragments.
     */
    private fun foreground(gray: Mat): Mat {
        val fg = Mat()
        subtractor.apply(gray, fg, config.learningRate)
        Imgproc.threshold(fg, fg, 200.0, 255.0, Imgproc.THRESH_BINARY) // drop MOG2 shadows
        Imgproc.morphologyEx(fg, fg, Imgproc.MORPH_OPEN, openKernel)
        Imgproc.dilate(fg, fg, dilateKernel)
        fgRatio = Core.countNonZero(fg).toDouble() / (fg.rows() * fg.cols())
        return fg
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
     * Without this the mesh slowly migrates onto the furniture and stays there: the
     * subject leaves, its points get handed to whatever texture is behind it.
     * The grace period is what lets a subject pause mid-gesture without being dropped.
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
     * Shi-Tomasi corners, gated by the absolute cornerMinEigenVal floor and, when the
     * foreground is usable, confined to it — so points land on the subject rather than
     * on whatever happens to be the most textured thing in the room.
     */
    private fun detect(gray: Mat, fg: Mat?) {
        if (config.maxPoints - points.size <= 0) return

        // Normalise to 0..1 so minEig is dimensionless and survives exposure changes.
        val gray32 = Mat()
        gray.convertTo(gray32, CvType.CV_32F, 1.0 / 255.0)
        val eig = Mat()
        Imgproc.cornerMinEigenVal(gray32, eig, 3, 3)
        gray32.release()
        eigMax = Core.minMaxLoc(eig).maxVal

        // Pass 1: the share reserved for the moving subject. Skin loses a straight
        // contest against furniture, so it gets its own budget before the room competes.
        val before = points.size
        if (fg != null && config.fgBias > 0.0) {
            val quota = ((config.maxPoints - points.size) * config.fgBias).toInt()
            if (quota > 0) seed(gray, eig, fg, quota)
        }
        seededFg = points.size - before

        // Pass 2: fill whatever is left from the whole frame, so the web still covers
        // the scene instead of collapsing onto the subject alone.
        val afterFg = points.size
        val rest = config.maxPoints - points.size
        if (rest > 0) seed(gray, eig, null, rest)
        seededAll = points.size - afterFg

        eig.release()
    }

    /** Reuses one bitmap; matToBitmap turns the 8UC1 mask into something Canvas can draw. */
    private fun maskToBitmap(fg: Mat): android.graphics.Bitmap {
        val bmp = maskBitmap ?: android.graphics.Bitmap.createBitmap(
            fg.cols(), fg.rows(), android.graphics.Bitmap.Config.ARGB_8888
        ).also { maskBitmap = it }
        org.opencv.android.Utils.matToBitmap(fg, bmp)
        return bmp
    }

    /** One Shi-Tomasi pass inside `region` (null = whole frame), gated by the eig floor. */
    private fun seed(gray: Mat, eig: Mat, region: Mat?, count: Int) {
        val mask = region?.clone() ?: Mat(gray.size(), CvType.CV_8UC1, Scalar(255.0))
        // Mask out ground already held by live points so re-detection doesn't stack
        // duplicate IDs on the same texture.
        for (p in points) {
            Imgproc.circle(mask, p, config.minDistance.toInt(), Scalar(0.0), -1)
        }

        val corners = MatOfPoint()
        Imgproc.goodFeaturesToTrack(
            gray, corners, count, QUALITY_LEVEL, config.minDistance, mask, 7
        )
        mask.release()

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
    private fun motionBoxes(fg: Mat): List<RectF> {
        val work = fg.clone() // findContours mutates its input
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(
            work, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
        )
        hierarchy.release()
        work.release()

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
        return boxes
    }

    /**
     * Web lines, recomputed from scratch every frame — nothing cached from the previous
     * layout, so a line follows whatever its two endpoints did. Each point keeps only its
     * `maxLinks` nearest neighbours inside `maxEdge`; a plain radius cutoff turns dense
     * texture into a solid blanket of lines.
     *
     * Returns flat [x0,y0,x1,y1,...] in analysis-frame coordinates.
     * ponytail: O(n^2) over <=150 points is ~11k checks; spatial hash if the cap rises.
     */
    private fun linkNearby(): FloatArray {
        val n = points.size
        if (n < 2) return FloatArray(0)

        val out = ArrayList<Float>()
        val seen = HashSet<Long>()
        val candidates = ArrayList<Pair<Double, Int>>(n)

        for (i in 0 until n) {
            candidates.clear()
            val a = points[i]
            for (j in 0 until n) {
                if (i == j) continue
                val b = points[j]
                val d = hypot(a.x - b.x, a.y - b.y)
                if (d <= config.maxEdge) candidates.add(d to j)
            }
            candidates.sortBy { it.first }

            for (k in 0 until minOf(config.maxLinks, candidates.size)) {
                val j = candidates[k].second
                val key = if (i < j) i.toLong() * n + j else j.toLong() * n + i
                if (!seen.add(key)) continue
                val b = points[j]
                out.add(a.x.toFloat()); out.add(a.y.toFloat())
                out.add(b.x.toFloat()); out.add(b.y.toFloat())
            }
        }
        return out.toFloatArray()
    }

    /** Scale everything from the analysis frame up to the rotated camera frame. */
    private fun buildFrame(
        lines: FloatArray,
        boxes: List<RectF>,
        outW: Int,
        outH: Int,
        mask: android.graphics.Bitmap?
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
        val scaledLines = FloatArray(lines.size) { k ->
            if (k % 2 == 0) lines[k] * sx else lines[k] * sy
        }
        val scaledBoxes = boxes.map {
            RectF(it.left * sx, it.top * sy, it.right * sx, it.bottom * sy)
        }
        return TrackedFrame(tracked, scaledLines, scaledBoxes, eigMax, fgRatio, mask)
    }
}
