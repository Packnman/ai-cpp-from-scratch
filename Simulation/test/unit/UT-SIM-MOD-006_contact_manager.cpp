#include "../TestSupport.hpp"
#include "simulation/MuJoCoPlant.hpp"

#include <algorithm>

using namespace ai::simulation;

int main() {
    simulation_test::Suite test{"SIM-MOD-006 Contact Manager"};
    MuJoCoPlant plant{simulation_test::model("test/simple_leg.xml")};
    plant.initialize();
    bool observed{};
    ContactState contact;
    for (int i = 0; i < 2'000 && !observed; ++i) {
        plant.step(0.001);
        const auto state = plant.getState();
        if (!state.contacts.empty()) {
            contact = state.contacts.front();
            observed = true;
        }
    }
    test.expect(observed, "UT-SIM-006-001",
                "falling simple leg produces floor contact");
    test.expect(contact.normalForce >= 0.0, "UT-SIM-006-002",
                "contact force is converted from contact frame");
    test.expect(!contact.bodyA.empty() && !contact.bodyB.empty(),
                "UT-SIM-006-003", "contact body pair uses logical names");
    return test.finish();
}
