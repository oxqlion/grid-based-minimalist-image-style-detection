import SwiftUI

/// Displays the classification result as a full-width banner that slides in from the bottom.
/// Auto-dismisses after 4 seconds.
struct ResultOverlayView: View {
    let result: ClassificationResult
    let onDismiss: () -> Void

    @State private var visible = false

    private var isMinimalist: Bool { result.isMinimalist }
    private var prob: Double { result.probability }

    var body: some View {
        VStack {
            Spacer()

            VStack(spacing: 0) {
                // Coloured indicator strip
                Rectangle()
                    .fill(isMinimalist ? Color(hue: 0.55, saturation: 0.6, brightness: 0.9)
                                       : Color(hue: 0.04, saturation: 0.75, brightness: 0.95))
                    .frame(height: 3)

                // Content
                HStack(spacing: 16) {
                    // Label + verdict
                    VStack(alignment: .leading, spacing: 4) {
                        Text(isMinimalist ? "MINIMALIST" : "NON-MINIMALIST")
                            .font(.system(size: 18, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white)
                            .tracking(1.5)

                        Text("confidence \(Int(prob * 100))%  ·  \(Int(result.inferenceMs))ms")
                            .font(.system(size: 11, weight: .regular, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.55))
                    }

                    Spacer()

                    // Big probability arc
                    ProbabilityArc(value: prob, isMinimalist: isMinimalist)
                        .frame(width: 52, height: 52)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 18)
                .background(.black.opacity(0.88))
            }
            .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
            .padding(.horizontal, 16)
            .padding(.bottom, 24)
            .offset(y: visible ? 0 : 120)
            .opacity(visible ? 1 : 0)
        }
        .onAppear {
            withAnimation(.spring(response: 0.45, dampingFraction: 0.78)) { visible = true }
            // Auto-dismiss
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                dismiss()
            }
        }
        .onTapGesture { dismiss() }
    }

    private func dismiss() {
        withAnimation(.easeIn(duration: 0.25)) { visible = false }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { onDismiss() }
    }
}

// MARK: - Probability Arc

private struct ProbabilityArc: View {
    let value: Double
    let isMinimalist: Bool

    private var color: Color {
        isMinimalist
            ? Color(hue: 0.55, saturation: 0.6, brightness: 0.9)
            : Color(hue: 0.04, saturation: 0.75, brightness: 0.95)
    }

    var body: some View {
        ZStack {
            Circle()
                .stroke(.white.opacity(0.08), lineWidth: 4)
            Circle()
                .trim(from: 0, to: value)
                .stroke(color, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.easeOut(duration: 0.6), value: value)
            Text("\(Int(value * 100))")
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundStyle(.white)
        }
    }
}
