import Foundation

enum Condition: String, CaseIterable, Identifiable {
    case amblyopia
    case amd
    case astigmatism
    case cataracts
    case colorBlindness = "color_blindness"
    case cvi
    case diabeticRetinopathy = "diabetic_retinopathy"

    var id: String { rawValue }

    var displayName: String {
        rawValue.replacingOccurrences(of: "_", with: " ")
            .capitalized
    }
}
