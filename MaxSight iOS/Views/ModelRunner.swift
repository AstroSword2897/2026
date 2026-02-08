import CoreML
import UIKit

class ModelRunner {
    static let shared = ModelRunner()

    private init() {}

    func runModel(on image: UIImage, condition: Condition) -> String {
        guard let model = ModelManager.shared.loadedModels[condition] else {
            return "Model not loaded"
        }

        // Convert UIImage to CVPixelBuffer
        guard let pixelBuffer = image.pixelBuffer() else { return "Invalid image" }

        do {
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)])
            let prediction = try model.prediction(from: input)

            return prediction.featureValue(for: "classLabel")?.stringValue ?? "No prediction"
        } catch {
            return "Inference failed"
        }
    }
}
