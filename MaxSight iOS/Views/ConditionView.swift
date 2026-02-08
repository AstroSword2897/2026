import SwiftUI

struct ConditionView: View {

    let condition: Condition

    var body: some View {
        VStack {
            Text("Condition View")
                .font(.headline)

            Image("example1")
                .resizable()
                .scaledToFit()
                .border(.gray)
                .overlay(
                    Text(condition.displayName)
                        .foregroundColor(.white)
                        .padding(6),
                    alignment: .topLeading
                )
        }
    }
}
