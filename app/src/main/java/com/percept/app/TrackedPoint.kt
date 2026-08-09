package com.percept.app

/** One tracked feature point, in the coordinate space of the rotated camera frame. */
data class TrackedPoint(
    val id: Int,
    val x: Float,
    val y: Float,
    /** px moved since the previous frame — real measured optical-flow displacement. */
    val speed: Float
)

/** Per-stage wall time for the last analysed frame, in ms. */
data class Timings(
    val convert: Float = 0f,
    val flow: Float = 0f,
    val detect: Float = 0f,
    val boxes: Float = 0f,
    val link: Float = 0f
) {
    val total get() = convert + flow + detect + boxes + link
}

data class TrackedFrame(
    val points: List<TrackedPoint>,
    /**
     * Web edges as index pairs [i0,j0,i1,j1,...] into `points`, NOT baked coordinates.
     * The overlay redraws faster than CV produces frames, so it needs to rebuild the
     * lines from wherever the points are being drawn right now.
     */
    val edges: IntArray,
    val boxes: List<android.graphics.RectF>,
    /**
     * Links between detected objects, as index pairs into `boxes`. This is the headline
     * relationship in the reference look: two people walking, a line drawn between them.
     * Point-level edges cannot express it — they only ever connect texture inside one
     * object, never one object to another.
     */
    val boxEdges: IntArray,
    val eigMax: Double,
    /** Fraction of the frame MOG2 called moving — near 1.0 means the camera itself moved. */
    val fgRatio: Double,
    val timings: Timings = Timings()
)
