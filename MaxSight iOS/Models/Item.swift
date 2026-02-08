import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date

    init(timestamp: Date = Date()) {
        self.timestamp = timestamp
    }
}
