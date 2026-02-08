import SwiftUI

func openImagesV6Names() -> [String] {
    guard let url = Bundle.main.url(forResource: "manifest", withExtension: "txt", subdirectory: "open_images_v6"),
          let raw = try? String(contentsOf: url, encoding: .utf8) else { return [] }
    return raw.split(separator: "\n").map { String($0).replacingOccurrences(of: ".jpg", with: "") }
}
