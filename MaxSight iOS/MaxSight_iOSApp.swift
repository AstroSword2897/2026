import SwiftUI
import SwiftData

@main
struct MaxSight_iOSApp: App {
    var sharedModelContainer: ModelContainer = {
        let schema = Schema([Item.self])
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
        do {
            return try ModelContainer(for: schema, configurations: [config])
        } catch {
            fatalError("ModelContainer error: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .edgesIgnoringSafeArea(.all) // fills entire screen without shifting up
        }
        .modelContainer(sharedModelContainer)
    }
}
