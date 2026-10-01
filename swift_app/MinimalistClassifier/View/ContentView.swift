import SwiftUI

struct ContentView: View {
    @StateObject private var vm = ContentViewModel()

    var body: some View {
        ZStack {
            // ── Background ─────────────────────────────────────────────────
            Color.black.ignoresSafeArea()

            // ── Camera Preview ─────────────────────────────────────────────
            CameraPreviewView(session: vm.camera.session)
                .ignoresSafeArea()

            // ── Feature Panel (bottom-left) ────────────────────────────────
            VStack {
                Spacer()
                HStack(alignment: .bottom) {
                    if let snapshot = vm.featureSnapshot {
                        FeaturePanelView(snapshot: snapshot)
                            .frame(width: 200)
                            .transition(.opacity.combined(with: .scale(scale: 0.95, anchor: .bottomLeading)))
                    }
                    Spacer()
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 120)  // sit above the capture button
            }
            .animation(.easeOut(duration: 0.3), value: vm.featureSnapshot != nil)

            // ── Shutter + Status Bar ───────────────────────────────────────
            VStack {
                Spacer()
                captureBar
                    .padding(.bottom, 36)
            }

            // ── Processing Spinner ─────────────────────────────────────────
            if vm.isProcessing {
                processingIndicator
            }

            // ── Result overlay ─────────────────────────────────────────────
            if vm.showResult, let result = vm.classificationResult {
                ResultOverlayView(result: result) {
                    vm.showResult = false
                }
            }

            // ── Error Alert ────────────────────────────────────────────────
            if let err = vm.errorMessage {
                errorBanner(err)
            }

            // ── Camera denied message ──────────────────────────────────────
            if !vm.camera.isAuthorized {
                cameraUnavailableView
            }
        }
        .preferredColorScheme(.dark)
    }

    // MARK: - Subviews

    /// Capture button + small status text.
    private var captureBar: some View {
        VStack(spacing: 10) {
            // Status label
            if vm.isProcessing {
                Label("Analysing…", systemImage: "sparkles")
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.7))
                    .transition(.opacity)
            }

            // Shutter button
            Button(action: vm.capturePhoto) {
                ZStack {
                    Circle()
                        .fill(.white.opacity(0.15))
                        .frame(width: 72, height: 72)
                    Circle()
                        .stroke(.white, lineWidth: 3)
                        .frame(width: 72, height: 72)
                    Circle()
                        .fill(.white)
                        .frame(width: 58, height: 58)
                        .scaleEffect(vm.isProcessing ? 0.6 : 1.0)
                        .animation(.easeInOut(duration: 0.25), value: vm.isProcessing)
                }
            }
            .disabled(vm.isProcessing || !vm.camera.isAuthorized)
        }
    }

    private var processingIndicator: some View {
        VStack(spacing: 8) {
            ProgressView()
                .tint(.white)
                .scaleEffect(1.2)
            Text("EXTRACTING FEATURES")
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white.opacity(0.6))
                .tracking(2)
        }
        .padding(20)
        .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 16))
    }

    private func errorBanner(_ message: String) -> some View {
        VStack {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                Text(message)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.white)
                Spacer()
                Button { vm.errorMessage = nil } label: {
                    Image(systemName: "xmark")
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
            .padding(14)
            .background(.black.opacity(0.85), in: RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal, 16)
            .padding(.top, 60)
            Spacer()
        }
    }

    private var cameraUnavailableView: some View {
        VStack(spacing: 12) {
            Image(systemName: "camera.slash")
                .font(.system(size: 40))
                .foregroundStyle(.white.opacity(0.4))
            Text("Camera access required.\nEnable it in Settings.")
                .font(.system(size: 14, design: .monospaced))
                .foregroundStyle(.white.opacity(0.5))
                .multilineTextAlignment(.center)
        }
    }
}

#Preview {
    ContentView()
}
