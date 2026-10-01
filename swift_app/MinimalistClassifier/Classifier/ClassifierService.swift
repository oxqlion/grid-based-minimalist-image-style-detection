import CoreML
import UIKit

// MARK: - Result

struct ClassificationResult {
    let isMinimalist: Bool
    let probability: Double    // P(Minimalist)
    let features: FeatureVector
    let inferenceMs: Double    // wall-clock time for extract + predict
}

// MARK: - Service

/// Wraps `MinimalistClassifier.mlmodel` and the preprocessing pipeline.
///
/// Usage:
///   let result = try await ClassifierService.shared.classify(image)
///
/// The `.mlmodel` handles StandardScaler + Random Forest internally,
/// so we only pass the raw 49-feature vector.
final class ClassifierService {

    static let shared = ClassifierService()

    // Lazily loaded Core ML model
    private let model: MLModel

    private init() {
        // ⚠️ Add MinimalistClassifier.mlmodel to your Xcode project target.
        // Xcode auto-generates a Swift class with the same name.
        guard let url = Bundle.main.url(forResource: "MinimalistClassifier", withExtension: "mlmodelc")
                     ?? Bundle.main.url(forResource: "MinimalistClassifier", withExtension: "mlmodel") else {
            fatalError("MinimalistClassifier.mlmodel not found in bundle. Add it to the Xcode target.")
        }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            model = try MLModel(contentsOf: url, configuration: config)
        } catch {
            fatalError("Failed to load Core ML model: \(error)")
        }
    }

    // MARK: - Public API

    func classify(_ image: UIImage) async throws -> ClassificationResult {
        let start = Date()

        // -- Extract features (CPU-bound, runs on calling task's executor)
        guard let features = ImagePreprocessor.extract(from: image) else {
            throw ClassifierError.preprocessingFailed
        }

        // -- Build MLFeatureProvider from FeatureVector
        let provider = try features.asMLFeatureProvider()

        // -- Run inference
        let output = try await model.prediction(from: provider)

        // -- Read outputs
        guard
            let labelValue = output.featureValue(for: "is_minimalist")?.int64Value,
            let probDict   = output.featureValue(for: "classProbability")?.dictionaryValue as? [Int64: Double]
        else {
            throw ClassifierError.outputParseFailed
        }

        let prob = probDict[1] ?? 0.0   // P(class=1 = Minimalist)
        let elapsed = Date().timeIntervalSince(start) * 1000

        return ClassificationResult(
            isMinimalist: labelValue == 1,
            probability:  prob,
            features:     features,
            inferenceMs:  elapsed
        )
    }
}

// MARK: - Errors

enum ClassifierError: LocalizedError {
    case preprocessingFailed
    case outputParseFailed

    var errorDescription: String? {
        switch self {
        case .preprocessingFailed: return "Failed to extract features from image."
        case .outputParseFailed:   return "Could not parse Core ML model output."
        }
    }
}

// MARK: - FeatureVector → MLFeatureProvider

extension FeatureVector {
    /// Converts the feature vector into an `MLDictionaryFeatureProvider`.
    /// Input names must exactly match the Core ML model's declared inputs.
    func asMLFeatureProvider() throws -> MLFeatureProvider {
        let dict: [String: MLFeatureValue] = [
            "photo_width":    .init(double: photo_width),
            "photo_height":   .init(double: photo_height),
            "aspect_ratio":   .init(double: aspect_ratio),
            "download_count": .init(double: download_count),
            "brightness":     .init(double: brightness),
            "contrast":       .init(double: contrast),
            "sharpness":      .init(double: sharpness),
            "edge_density":   .init(double: edge_density),
            "saturation":     .init(double: saturation),
            "colorfulness":   .init(double: colorfulness),
            "texture_ent":    .init(double: texture_ent),
            "hue_std":        .init(double: hue_std),
            "edge_1":  .init(double: edge_1),  "sharp_1":  .init(double: sharp_1),  "bright_1":  .init(double: bright_1),
            "edge_2":  .init(double: edge_2),  "sharp_2":  .init(double: sharp_2),  "bright_2":  .init(double: bright_2),
            "edge_3":  .init(double: edge_3),  "sharp_3":  .init(double: sharp_3),  "bright_3":  .init(double: bright_3),
            "edge_4":  .init(double: edge_4),  "sharp_4":  .init(double: sharp_4),  "bright_4":  .init(double: bright_4),
            "edge_5":  .init(double: edge_5),  "sharp_5":  .init(double: sharp_5),  "bright_5":  .init(double: bright_5),
            "edge_6":  .init(double: edge_6),  "sharp_6":  .init(double: sharp_6),  "bright_6":  .init(double: bright_6),
            "edge_7":  .init(double: edge_7),  "sharp_7":  .init(double: sharp_7),  "bright_7":  .init(double: bright_7),
            "edge_8":  .init(double: edge_8),  "sharp_8":  .init(double: sharp_8),  "bright_8":  .init(double: bright_8),
            "edge_9":  .init(double: edge_9),  "sharp_9":  .init(double: sharp_9),  "bright_9":  .init(double: bright_9),
            "edge_max_ratio":   .init(double: edge_max_ratio),
            "edge_std":         .init(double: edge_std),
            "empty_ratio":      .init(double: empty_ratio),
            "center_ratio":     .init(double: center_ratio),
            "sharp_max_ratio":  .init(double: sharp_max_ratio),
            "sharp_std":        .init(double: sharp_std),
            "thirds_ratio":     .init(double: thirds_ratio),
            "vertical_balance": .init(double: vertical_balance),
            "horiz_balance":    .init(double: horiz_balance),
            "bright_std":       .init(double: bright_std),
        ]
        return try MLDictionaryFeatureProvider(dictionary: dict)
    }
}

// Convenience init for Double on MLFeatureValue (avoids verbose syntax)
private extension MLFeatureValue {
    convenience init(double: Double) {
        self.init(double: double)
    }
}
