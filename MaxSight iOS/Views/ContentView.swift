import SwiftUI
import UIKit
import CoreML

struct ContentView: View {
    @State private var selectedCondition: Condition = .amblyopia
    @State private var selectedImage: String = "example1"
    @State private var predictionText: String = "Loading..."
    @State private var modelsLoaded = false

    let datasetImages = openImagesV6Names()

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 12) {

                // MARK: - Top Controls
                VStack(spacing: 8) {
                    Picker("Condition", selection: $selectedCondition) {
                        ForEach(Condition.allCases) { Text($0.displayName).tag($0) }
                    }
                    .pickerStyle(.segmented)

                    Picker("Dataset Image", selection: $selectedImage) {
                        ForEach(datasetImages, id: \.self) { Text($0) }
                    }
                    .pickerStyle(.menu)

                    Button("Reset Image") {
                        if let random = datasetImages.randomElement() {
                            selectedImage = random
                            runModel()
                        }
                    }
                    .disabled(!modelsLoaded) // disable until models are ready
                }
                .padding(.horizontal)

                // MARK: - Panels
                VStack(spacing: 8) {
                    PanelView(title: "Reality View", imageName: selectedImage, overlayText: nil)
                    PanelView(title: "Condition View", imageName: selectedImage, overlayText: nil)
                    PanelView(title: "Awareness View", imageName: selectedImage, overlayText: predictionText)
                }
                .frame(height: geometry.size.height * 0.75)
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
        .onAppear {
            preloadModels()
        }
        .onChange(of: selectedImage) { _ in runModel() }
        .onChange(of: selectedCondition) { _ in runModel() }
    }

    // MARK: - Preload All Models
    func preloadModels() {
        DispatchQueue.global(qos: .userInitiated).async {
            for condition in Condition.allCases {
                ModelManager.shared.loadModel(for: condition)
            }
            DispatchQueue.main.async {
                modelsLoaded = true
                runModel()
            }
        }
    }

    // MARK: - Run CoreML Model
    func runModel() {
        guard modelsLoaded else {
            predictionText = "Models loading..."
            return
        }

        guard let uiImage = UIImage(named: selectedImage) else {
            predictionText = "Image not found"
            return
        }

        let resizedImage = uiImage.resized(to: CGSize(width: 224, height: 224)) // adjust size to your model
        predictionText = ModelRunner.shared.runModel(on: resizedImage, condition: selectedCondition)
    }
}

// MARK: - Reusable Panel View
struct PanelView: View {
    let title: String
    let imageName: String
    let overlayText: String?

    var body: some View {
        VStack {
            Text(title).font(.headline)
            Image(imageName)
                .resizable()
                .scaledToFit()
                .border(Color.gray)
                .overlay(
                    overlayText.map {
                        Text($0)
                            .foregroundColor(.white)
                            .padding(6)
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(6)
                            .padding(8)
                    },
                    alignment: .bottomLeading
                )
        }
        .background(Color(white: 0.95))
        .cornerRadius(8)
        .frame(maxWidth: .infinity)
    }
}

// MARK: - UIImage Resizing Extension
extension UIImage {
    func resized(to size: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized ?? self
    }
}
