package com.percept.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private val analysisExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()
    private lateinit var config: TrackerConfig

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        OpenCVLoader.initLocal()

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)

        // Tuning comes from the launch intent so the room can be calibrated over adb
        // without a rebuild. See TrackerConfig.
        config = TrackerConfig.fromIntent(intent)
        overlayView.labelSpeed = config.labelSpeed.toFloat()
        overlayView.smoothing = config.smoothing.toFloat()
        overlayView.trailLength = config.trailLength
        overlayView.fastSpeed = config.fastSpeed.toFloat()
        android.util.Log.i("Percept", "config = $config")

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    // CV runs off the UI thread; OverlayView.update posts the invalidate.
                    val tracker = TrackerAnalyzer(config) { frame, w, h ->
                        overlayView.update(frame, w, h)
                    }
                    // Tap isolates one layer at a time; all numeric diagnostics go to
                    // logcat rather than on top of the image.
                    overlayView.setOnClickListener { overlayView.cycleMode() }
                    it.setAnalyzer(analysisExecutor, tracker)
                }

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }
}
