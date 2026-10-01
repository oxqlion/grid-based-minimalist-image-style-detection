import SwiftUI

// MARK: - Feature Panel

/// A compact overlay showing 6 key feature values as labelled horizontal bars.
/// Rendered over the camera preview — uses semi-transparent glass background.
struct FeaturePanelView: View {
    let snapshot: ContentViewModel.FeatureSnapshot

    // Each feature is displayed with a label and a bar clamped to [0,1].
    // colorfulness is in [0,~100] so we normalise it to a reasonable max.
    private var items: [(label: String, value: Double, max: Double)] {
        [
            ("BRIGHTNESS",   snapshot.brightness,            1.0),
            ("CONTRAST",     snapshot.contrast,              0.5),   // std rarely > 0.5
            ("EDGE DENSITY", snapshot.edgeDensity,           1.0),
            ("SATURATION",   snapshot.saturation,            1.0),
            ("COLORFULNESS", snapshot.colorfulness / 80.0,   1.0),   // normalise ~[0,80]
            ("EMPTY RATIO",  snapshot.emptyRatio,            1.0),
        ]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("FEATURES")
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white.opacity(0.5))
                .tracking(2)

            ForEach(items, id: \.label) { item in
                FeatureRow(
                    label: item.label,
                    value: item.value,
                    normalised: min(1, max(0, item.value / item.max))
                )
            }
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(.white.opacity(0.1), lineWidth: 0.5)
        )
    }
}

// MARK: - Feature Row

private struct FeatureRow: View {
    let label: String
    let value: Double
    let normalised: Double   // [0,1] for bar width

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(label)
                    .font(.system(size: 8, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.6))
                    .tracking(1)
                Spacer()
                Text(String(format: "%.4f", value))
                    .font(.system(size: 8, weight: .regular, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.8))
            }

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    // Track
                    RoundedRectangle(cornerRadius: 2)
                        .fill(.white.opacity(0.08))
                        .frame(height: 3)
                    // Fill
                    RoundedRectangle(cornerRadius: 2)
                        .fill(barColor(for: normalised))
                        .frame(width: geo.size.width * normalised, height: 3)
                }
            }
            .frame(height: 3)
        }
    }

    private func barColor(for n: Double) -> Color {
        // Gradient from cool blue → warm amber depending on value
        if n < 0.33 { return Color(hue: 0.58, saturation: 0.6, brightness: 0.9) }
        if n < 0.66 { return Color(hue: 0.45, saturation: 0.7, brightness: 0.9) }
        return Color(hue: 0.10, saturation: 0.8, brightness: 1.0)
    }
}

// MARK: - Preview

#Preview {
    ZStack {
        Color.black.ignoresSafeArea()
        FeaturePanelView(snapshot: .init(
            brightness:   0.42,
            contrast:     0.19,
            edgeDensity:  0.13,
            saturation:   0.31,
            colorfulness: 35.6,
            emptyRatio:   0.33
        ))
        .padding()
    }
}
