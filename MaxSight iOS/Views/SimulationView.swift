import SwiftUI

struct SimulationView: View {

    var body: some View {
        VStack {
            Text("Simulation View")
                .font(.headline)

            Image("example2")
                .resizable()
                .scaledToFit()
                .border(.gray)
        }
    }
}
