import SwiftUI
import AVFoundation

/// A SwiftUI wrapper around `AVCaptureVideoPreviewLayer`.
/// Displays the live camera feed inside a SwiftUI view hierarchy.
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.session = session
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}

    // MARK: - Backing UIView

    final class PreviewUIView: UIView {
        var session: AVCaptureSession? {
            didSet {
                guard let session else { return }
                previewLayer.session = session
            }
        }

        private var previewLayer: AVCaptureVideoPreviewLayer {
            // Safe cast: we set the layerClass below
            return layer as! AVCaptureVideoPreviewLayer
        }

        // Use AVCaptureVideoPreviewLayer as the view's backing CALayer
        override class var layerClass: AnyClass {
            AVCaptureVideoPreviewLayer.self
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer.videoGravity = .resizeAspectFill
            previewLayer.frame = bounds

            // Keep orientation in sync
            if let connection = previewLayer.connection,
               connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
        }
    }
}
