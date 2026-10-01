import AVFoundation
import UIKit
import Combine

/// Manages the AVCaptureSession lifecycle and photo capture.
/// Publishes a UIImage whenever a photo is successfully captured.
final class CameraManager: NSObject, ObservableObject {

    // MARK: - Published State

    @Published var capturedImage: UIImage? = nil
    @Published var isAuthorized: Bool = false
    @Published var error: String? = nil

    // MARK: - AVFoundation Objects

    let session = AVCaptureSession()
    private var photoOutput = AVCapturePhotoOutput()
    private var deviceInput: AVCaptureDeviceInput?

    // MARK: - Setup

    /// Call once on app start. Checks permission then configures the session.
    func configure() {
        checkPermission()
    }

    private func checkPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            isAuthorized = true
            setupSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.isAuthorized = granted
                    if granted { self?.setupSession() }
                }
            }
        default:
            DispatchQueue.main.async {
                self.error = "Camera access denied. Enable it in Settings."
            }
        }
    }

    private func setupSession() {
        session.beginConfiguration()
        session.sessionPreset = .photo

        // Add video input (back camera)
        guard
            let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: camera),
            session.canAddInput(input)
        else {
            DispatchQueue.main.async { self.error = "Could not access back camera." }
            session.commitConfiguration()
            return
        }
        session.addInput(input)
        deviceInput = input

        // Add photo output
        guard session.canAddOutput(photoOutput) else {
            DispatchQueue.main.async { self.error = "Could not configure photo output." }
            session.commitConfiguration()
            return
        }
        session.addOutput(photoOutput)
        photoOutput.isHighResolutionCaptureEnabled = true

        session.commitConfiguration()

        // Start session on a background thread
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
        }
    }

    // MARK: - Capture

    /// Triggers a photo capture. Result is published via `capturedImage`.
    func capturePhoto() {
        let settings = AVCapturePhotoSettings()
        settings.flashMode = .off
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension CameraManager: AVCapturePhotoCaptureDelegate {
    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        if let error = error {
            DispatchQueue.main.async { self.error = error.localizedDescription }
            return
        }
        guard
            let data = photo.fileDataRepresentation(),
            let image = UIImage(data: data)
        else { return }

        DispatchQueue.main.async {
            self.capturedImage = image
        }
    }
}
