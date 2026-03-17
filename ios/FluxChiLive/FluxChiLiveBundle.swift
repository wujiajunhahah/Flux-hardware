import WidgetKit
import SwiftUI

@main
struct FluxChiLiveBundle: WidgetBundle {
    var body: some Widget {
        FluxChiLiveActivity()
        FluxSessionCountWidget()
        FluxStaminaWidget()
        FluxTrendWidget()
        FluxDashboardWidget()
    }
}
