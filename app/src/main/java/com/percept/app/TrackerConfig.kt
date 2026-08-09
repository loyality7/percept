package com.percept.app

import android.content.Intent

/**
 * Tuning knobs. Every value can be overridden at launch from adb, so the camera can be
 * calibrated against the real room without a rebuild:
 *
 *   adb shell am start -n com.percept.app/.MainActivity --ef min_eig 0.004 --ei max_points 120
 *
 * (--ef = float extra, --ei = int extra. Omitted keys keep the defaults below.)
 */
data class TrackerConfig(
    /**
     * Absolute Shi-Tomasi floor. Its only job is to keep sensor noise in a dark or blank
     * frame from being promoted to "corner" — NOT to be selective about real texture.
     * Skin scores low (soft, low-contrast), so a high floor erases hands entirely.
     * Blank wall still shows points => raise. Hands show none => lower.
     */
    val minEig: Double = 0.003,
    val maxPoints: Int = 260,
    val minPoints: Int = 60,
    val redetectEvery: Int = 8,
    val minDistance: Double = 9.0,
    /** Forward-backward flow error, px. Too tight and fast-moving points get killed. */
    val fbThreshold: Double = 3.0,
    /** Longest web line, analysis-frame px. */
    val maxEdge: Double = 42.0,
    /** Max web lines per point — this is what keeps the mesh sparse instead of a blanket. */
    val maxLinks: Int = 3,
    val minBoxArea: Double = 250.0,
    /** px/frame a point must move before its ID is drawn. */
    val labelSpeed: Double = 1.0,

    /**
     * Fraction of each re-detect reserved for the moving subject, 0.0..1.0.
     *
     * Shi-Tomasi picks the *most textured* thing, and furniture beats skin every time, so
     * an unbiased detector spends all its points on the room. This reserves a share for
     * the MOG2 foreground and fills the remainder from the whole frame — the subject gets
     * covered without the rest of the web being thrown away. 0 = no bias, 1 = subject only
     * (which strips the mesh off everything else — usually not what you want).
     */
    val fgBias: Double = 0.6,
    /** How fast MOG2 forgets. Lower = a subject that pauses stays foreground longer. */
    val learningRate: Double = 0.004,
    /** 1 = retire points that sit on static background. Off by default: it thins the web. */
    val pruneBg: Int = 0,
    /** Frames a tracked point may sit on background before it is dropped (needs pruneBg=1). */
    val bgGrace: Int = 20,
    /** Above this foreground fraction the camera itself moved — the bias is skipped. */
    val fgMaxRatio: Double = 0.55
) {
    companion object {
        fun fromIntent(intent: Intent?): TrackerConfig {
            val d = TrackerConfig()
            if (intent == null) return d
            return TrackerConfig(
                minEig = intent.getFloatExtra("min_eig", d.minEig.toFloat()).toDouble(),
                maxPoints = intent.getIntExtra("max_points", d.maxPoints),
                minPoints = intent.getIntExtra("min_points", d.minPoints),
                redetectEvery = intent.getIntExtra("redetect_every", d.redetectEvery),
                minDistance = intent.getFloatExtra("min_distance", d.minDistance.toFloat()).toDouble(),
                fbThreshold = intent.getFloatExtra("fb_threshold", d.fbThreshold.toFloat()).toDouble(),
                maxEdge = intent.getFloatExtra("max_edge", d.maxEdge.toFloat()).toDouble(),
                maxLinks = intent.getIntExtra("max_links", d.maxLinks),
                minBoxArea = intent.getFloatExtra("min_box_area", d.minBoxArea.toFloat()).toDouble(),
                labelSpeed = intent.getFloatExtra("label_speed", d.labelSpeed.toFloat()).toDouble(),
                fgBias = intent.getFloatExtra("fg_bias", d.fgBias.toFloat()).toDouble(),
                learningRate = intent.getFloatExtra("learning_rate", d.learningRate.toFloat()).toDouble(),
                pruneBg = intent.getIntExtra("prune_bg", d.pruneBg),
                bgGrace = intent.getIntExtra("bg_grace", d.bgGrace),
                fgMaxRatio = intent.getFloatExtra("fg_max_ratio", d.fgMaxRatio.toFloat()).toDouble()
            )
        }
    }
}
