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
 */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    @Volatile private var frame: TrackedFrame? = null
    private var frameW = 1
    private var frameH = 1
    private var scaledLines = FloatArray(0)

    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(120, 200, 200, 200)
        strokeWidth = 1.2f
    }
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(220, 255, 40, 40)
        style = Paint.Style.STROKE
        strokeWidth = 2f
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
    private var fps = 0.0

    /** px/frame before a point earns its ID label; set from TrackerConfig. */
    var labelSpeed = 0.6f

    /**
     * Debug layers, cycled by tapping the screen. Isolating one layer at a time is the
     * only way to tell "the detector is wrong" from "the linker is wrong" — with
     * everything drawn at once a bad layer just looks like general noise.
     */
    var mode = MODE_ALL
        private set

    fun cycleMode(): Int {
        mode = (mode + 1) % MODE_NAMES.size
        postInvalidate()
        return mode
    }

    private val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        alpha = 110
        isFilterBitmap = false
    }
    private val maskDst = android.graphics.RectF()

    companion object {
        const val MODE_ALL = 0
        const val MODE_MASK = 1      // what MOG2 thinks is moving — the detector's input
        const val MODE_POINTS = 2    // detection + tracking only
        const val MODE_LINES = 3     // the web alone
        const val MODE_BOXES = 4     // contour boxes alone
        val MODE_NAMES = arrayOf("ALL", "MASK", "POINTS", "LINES", "BOXES")
    }

    fun update(newFrame: TrackedFrame, srcW: Int, srcH: Int) {
        frame = newFrame
        frameW = srcW
        frameH = srcH
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val f = frame ?: return
        val sx = width.toFloat() / frameW
        val sy = height.toFloat() / frameH

        // MASK mode paints the raw MOG2 foreground the detector was seeded from. White
        // means "the pipeline believes this moved" — if your hand is not white here, no
        // amount of tuning downstream will put points on it.
        f.maskPreview?.takeIf { mode == MODE_MASK }?.let {
            maskDst.set(0f, 0f, width.toFloat(), height.toFloat())
            canvas.drawBitmap(it, null, maskDst, maskPaint)
        }

        // Whole web in one batched drawLines call rather than hundreds of drawLine ops.
        if (mode == MODE_ALL || mode == MODE_MASK || mode == MODE_LINES) {
            if (scaledLines.size != f.lines.size) scaledLines = FloatArray(f.lines.size)
            for (k in f.lines.indices) {
                scaledLines[k] = if (k % 2 == 0) f.lines[k] * sx else f.lines[k] * sy
            }
            if (scaledLines.isNotEmpty()) canvas.drawLines(scaledLines, linePaint)
        }

        if (mode == MODE_ALL || mode == MODE_MASK || mode == MODE_BOXES) {
            for (box in f.boxes) {
                canvas.drawRect(
                    box.left * sx, box.top * sy, box.right * sx, box.bottom * sy, boxPaint
                )
            }
        }

        // Only label points that actually moved this frame. Keeps the HUD readable, and
        // makes the IDs self-evidently real: a number rides its point while it travels.
        var movers = 0
        val drawPoints = mode == MODE_ALL || mode == MODE_MASK || mode == MODE_POINTS
        for (p in f.points) {
            if (p.speed >= labelSpeed) movers++
            if (!drawPoints) continue
            val x = p.x * sx
            val y = p.y * sy
            canvas.drawCircle(x, y, 2.5f, pointPaint)
            if (p.speed >= labelSpeed) {
                canvas.drawText(p.id.toString(), x + 5f, y - 5f, idPaint)
            }
        }

        val now = System.nanoTime()
        if (lastDrawNanos != 0L) {
            fps = 0.9 * fps + 0.1 * (1e9 / (now - lastDrawNanos).coerceAtLeast(1))
        }
        lastDrawNanos = now

        // "moving" is the honesty check: it should climb when something in frame moves
        // and fall to ~0 when everything is still.
        // fg% is the diagnostic that matters: it should sit low with a still camera and
        // spike toward 100 the moment you move the phone (which makes masking useless).
        canvas.drawText(
            "pts %d  mov %d  lines %d  box %d  fg %.0f%%  eig %.3f  %.0ffps"
                .format(
                    f.points.size, movers, f.lines.size / 4, f.boxes.size,
                    f.fgRatio * 100, f.eigMax, fps
                ),
            16f, 48f, hudPaint
        )
        canvas.drawText("tap: ${MODE_NAMES[mode]}", 16f, 80f, hudPaint)
    }
}
