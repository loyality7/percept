package com.percept.app

/** One tracked feature point, in the coordinate space of the rotated camera frame. */
data class TrackedPoint(
    val id: Int,
    val x: Float,
    val y: Float,
    /** px moved since the previous frame — real measured optical-flow displacement. */
    val speed: Float
)

data class TrackedFrame(
    val points: List<TrackedPoint>,
    /** Flat [x0,y0,x1,y1, ...] so the overlay can draw the whole web in one batched call. */
    val lines: FloatArray,
    val boxes: List<android.graphics.RectF>,
    val eigMax: Double,
    /** Fraction of the frame MOG2 called moving — near 1.0 means the camera itself moved. */
    val fgRatio: Double,
    /** The actual MOG2 foreground the detector was seeded from. Only built in MASK mode. */
    val maskPreview: android.graphics.Bitmap? = null
)
