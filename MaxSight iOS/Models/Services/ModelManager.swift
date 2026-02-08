import Foundation
import CoreML
import UIKit
import SwiftUI
import Combine


class ModelManager: ObservableObject {

    static let shared = ModelManager()

    @Published var loadedModels: [Condition: MLModel] = [:]

    private init() {}

    func loadModel(for condition: Condition) {
        if loadedModels[condition] != nil { return }

        guard let url = Bundle.main.url(
            forResource: condition.rawValue,
            withExtension: "mlmodelc"
        ) else {
            print("Model not found:", condition.rawValue)
            return
        }

        do {
            let model = try MLModel(contentsOf: url)
            loadedModels[condition] = model
        } catch {
            print("Model load failed:", error)
        }
    }

    func runInference(
        image: UIImage,
        condition: Condition
    ) -> UIImage {
        return image // placeholder
    }
}
