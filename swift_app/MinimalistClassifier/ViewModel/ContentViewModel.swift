import SwiftUI
import Combine

@MainActor
final class ContentViewModel: ObservableObject {

    // MARK: - Dependencies
    let camera = CameraManager()

    // MARK: - State
    @Published var classificationResult: ClassificationResult? = nil
    @Published var isProcessing: Bool = false
    @Published var errorMessage: String? = nil
    @Published var showResult: Bool = false

    // MARK: - Displayed feature snapshots (for the feature panel)
    // We surface a small hand-picked subset of global features for the overlay.
    struct FeatureSnapshot {
        let brightness:   Double
        let contrast:     Double
        let edgeDensity:  Double
        let saturation:   Double
        let colorfulness: Double
        let emptyRatio:   Double
    }
    @Published var featureSnapshot: FeatureSnapshot? = nil

    private var cancellables = Set<AnyCancellable>()

    // MARK: - Init

    init() {
        camera.configure()

        // When camera captures an image → run the full pipeline
        camera.$capturedImage
            .compactMap { $0 }
            .sink { [weak self] image in
                Task { await self?.runInference(on: image) }
            }
            .store(in: &cancellables)
    }

    // MARK: - Actions

    func capturePhoto() {
        guard !isProcessing else { return }
        camera.capturePhoto()
    }

    // MARK: - Pipeline

    private func runInference(on image: UIImage) async {
        isProcessing = true
        errorMessage = nil

        do {
            let result = try await ClassifierService.shared.classify(image)

            // Publish result
            classificationResult = result
            featureSnapshot = FeatureSnapshot(
                brightness:   result.features.brightness,
                contrast:     result.features.contrast,
                edgeDensity:  result.features.edge_density,
                saturation:   result.features.saturation,
                colorfulness: result.features.colorfulness,
                emptyRatio:   result.features.empty_ratio
            )
            showResult = true
        } catch {
            errorMessage = error.localizedDescription
        }

        isProcessing = false
    }
}
