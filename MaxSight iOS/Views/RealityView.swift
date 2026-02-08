import SwiftUI

struct RealityView: View {

    var body: some View {
        VStack {
            Text("Reality View")
                .font(.headline)

            Image("example1")
                .resizable()
                .scaledToFit()
                .border(.gray)
        }
    }
}
