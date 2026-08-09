package com.percept.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/**
 * Draws the machine-vision HUD. One colour per layer so each can be verified
 * independently: points white, web gray, motion boxes red, IDs cyan.
 *
 * Rendering is decoupled from analysis. CV produces frames at ~15-25fps; this view
 * redraws every display frame and eases each point toward its latest measured position.
 * Without that, the overlay steps in visible jumps whenever the phone moves — the single
 * biggest difference between "smooth demo" and "stuttery demo".
 *
 * The interpolation only ever moves a point toward a real measurement. No position is
 * invented, extrapolated past the last reading, or held alive after its point dies.
 */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    @Volatile private var frame: TrackedFrame? = null
    private var frameW = 1
    private var frameH = 1
    private var lineBuf = FloatArray(0)

    /** id -> currently displayed position, eased toward the measured one each draw. */
    private val shown = HashMap<Int, FloatArray>()
    private val scratch = HashMap<Int, FloatArray>()

    /**
     * id -> recent measured positions, newest last. Sampled once per CV frame, not per
     * draw, so the trail covers a real span of time rather than a few display frames.
     * Every entry is somewhere the point genuinely was.
     */
    private val trails = HashMap<Int, ArrayDeque<FloatArray>>()
    private val trailScratch = HashMap<Int, ArrayDeque<FloatArray>>()
    private var trailBuf = FloatArray(0)
    /** Set when a new CV frame lands; the draw loop consumes it to advance the trails. */
    private var frameIsNew = false

    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(150, 225, 225, 225)
        strokeWidth = 1.7f
    }
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(220, 255, 40, 40)
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }
    private val boxLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(240, 255, 80, 80)
        textSize = 26f
    }
    private val idPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(200, 0, 255, 255)
        textSize = 18f
    }
    private val hudPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        textSize = 28f
    }

    private var lastDrawNanos = 0L
    private var renderFps = 0.0
    private var lastFrameNanos = 0L
    private var cvFps = 0.0

    /** px/frame before a point earns its ID label; set from TrackerConfig. */
    var labelSpeed = 0.6f

    /** 0..1 per display frame. Higher snaps harder to the measurement, lower is smoother. */
    var smoothing = 0.35f

    /** Past positions kept per point; 0 disables trails. */
    var trailLength = 7

    /** px/frame at which a point reaches full "fast" weight. */
    var fastSpeed = 3.0f

    private val trailPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(70, 255, 255, 255)
        strokeWidth = 1.2f
    }

    /**
     * Debug layers, cycled by tapping the screen. Isolating one layer at a time is the
     * only way to tell "the detector is wrong" from "the linker is wrong" — with
     * everything drawn at once a bad layer just looks like general noise.
     */
    var mode = MODE_ALL
        private set

    fun cycleMode(): Int {
        mode = (mode + 1) % MODE_NAMES.size
        return mode
    }

    companion object {
        const val MODE_ALL = 0
        const val MODE_POINTS = 1    // detection + tracking only
        const val MODE_LINES = 2     // the web alone
        const val MODE_BOXES = 3     // objects and the links between them
        val MODE_NAMES = arrayOf("ALL", "POINTS", "LINES", "BOXES")
    }

    fun update(newFrame: TrackedFrame, srcW: Int, srcH: Int) {
        frame = newFrame
        frameW = srcW
        frameH = srcH

        val now = System.nanoTime()
        if (lastFrameNanos != 0L) {
            cvFps = 0.9 * cvFps + 0.1 * (1e9 / (now - lastFrameNanos).coerceAtLeast(1))
        }
        lastFrameNanos = now
        frameIsNew = true
        // No invalidate here: the draw loop below is already running at display rate.
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // Schedule the next display frame first, so the loop keeps running at ~60fps
        // regardless of how slowly CV is delivering.
        postInvalidateOnAnimation()

        val now = System.nanoTime()
        if (lastDrawNanos != 0L) {
            renderFps = 0.9 * renderFps + 0.1 * (1e9 / (now - lastDrawNanos).coerceAtLeast(1))
        }
        lastDrawNanos = now

        val f = frame ?: return
        val sx = width.toFloat() / frameW
        val sy = height.toFloat() / frameH

        // Ease every live point toward its measured position. Rebuilding the map from the
        // current frame's ids is also what retires dead points: an id that stopped being
        // reported simply never gets copied across.
        scratch.clear()
        for (p in f.points) {
            val cur = shown[p.id]
            val pos = if (cur == null) {
                floatArrayOf(p.x, p.y) // first sighting: start where it was measured
            } else {
                cur[0] += (p.x - cur[0]) * smoothing
                cur[1] += (p.y - cur[1]) * smoothing
                cur
            }
            scratch[p.id] = pos
        }
        shown.clear()
        shown.putAll(scratch)

        // Advance trails only when a new measurement arrived, so trail length is a span
        // of real time rather than of display frames. Rebuilding from the current ids
        // retires dead points here too.
        if (frameIsNew && trailLength > 0) {
            frameIsNew = false
            trailScratch.clear()
            for (p in f.points) {
                val q = trails[p.id] ?: ArrayDeque()
                q.addLast(floatArrayOf(p.x, p.y))
                while (q.size > trailLength) q.removeFirst()
                trailScratch[p.id] = q
            }
            trails.clear()
            trails.putAll(trailScratch)
        }

        if (trailLength > 0 && (mode == MODE_ALL || mode == MODE_POINTS)) {
            var need = 0
            for (q in trails.values) if (q.size > 1) need += (q.size - 1) * 4
            if (trailBuf.size != need) trailBuf = FloatArray(need)
            var w = 0
            for (q in trails.values) {
                if (q.size < 2) continue
                var prev: FloatArray? = null
                for (pt in q) {
                    if (prev != null) {
                        trailBuf[w++] = prev[0] * sx; trailBuf[w++] = prev[1] * sy
                        trailBuf[w++] = pt[0] * sx; trailBuf[w++] = pt[1] * sy
                    }
                    prev = pt
                }
            }
            if (w > 0) canvas.drawLines(trailBuf, 0, w, trailPaint)
        }

        // Web rebuilt from the interpolated positions, in one batched drawLines call.
        if (mode == MODE_ALL || mode == MODE_LINES) {
            val need = f.edges.size * 2
            if (lineBuf.size != need) lineBuf = FloatArray(need)
            var w = 0
            var e = 0
            while (e < f.edges.size) {
                val a = shown[f.points[f.edges[e]].id]
                val b = shown[f.points[f.edges[e + 1]].id]
                if (a != null && b != null) {
                    lineBuf[w++] = a[0] * sx; lineBuf[w++] = a[1] * sy
                    lineBuf[w++] = b[0] * sx; lineBuf[w++] = b[1] * sy
                }
                e += 2
            }
            if (w > 0) canvas.drawLines(lineBuf, 0, w, linePaint)
        }

        if (mode == MODE_ALL || mode == MODE_BOXES) {
            for (box in f.boxes) {
                val r = box.rect
                canvas.drawRect(r.left * sx, r.top * sy, r.right * sx, r.bottom * sy, boxPaint)
                // Label and tracking ID come from the ML detector; motion blobs have
                // neither, so nothing is drawn rather than a made-up name.
                val tag = listOfNotNull(box.label, box.trackingId?.let { "#$it" })
                    .joinToString(" ")
                if (tag.isNotEmpty()) {
                    canvas.drawText(tag, r.left * sx + 4f, r.top * sy - 6f, boxLabelPaint)
                }
            }
        }

        // Only label points that actually moved this frame. Keeps the HUD readable, and
        // makes the IDs self-evidently real: a number rides its point while it travels.
        var movers = 0
        val drawPoints = mode == MODE_ALL || mode == MODE_POINTS
        for (p in f.points) {
            if (p.speed >= labelSpeed) movers++
            if (!drawPoints) continue
            val pos = shown[p.id] ?: continue
            val x = pos[0] * sx
            val y = pos[1] * sy
            // Radius carries the real optical-flow magnitude, so fast motion reads
            // differently from slow without inventing anything: the number is already
            // measured every frame, it was simply being discarded.
            val fast = (p.speed / fastSpeed).coerceIn(0f, 1f)
            canvas.drawCircle(x, y, 2f + 2.2f * fast, pointPaint)
            if (p.speed >= labelSpeed) {
                canvas.drawText(p.id.toString(), x + 5f, y - 5f, idPaint)
            }
        }

        // fg% is the diagnostic that explains most surprises: it stays low with a still
        // camera and spikes toward 100 the moment the phone itself moves.
        // One line only. Per-stage timings and detection diagnostics go to logcat
        // (adb logcat -s Percept) — they belong in the console, not over the image.
        canvas.drawText(
            "pts %d  obj %d  links %d  %.0f/%.0ffps  %s"
                .format(
                    f.points.size, f.boxes.size, f.boxEdges.size / 2,
                    cvFps, renderFps, MODE_NAMES[mode]
                ),
            16f, 48f, hudPaint
        )
    }
}
