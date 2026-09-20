#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace actuator_test {

class Suite {
    public:
        explicit Suite(std::string_view name) : _name(name) {}

        void expect(bool condition, std::string_view id,
                    std::string_view description) {
            if (!condition) {
                std::cerr << "FAIL " << id << ": " << description << '\n';
                std::exit(1);
            }
            ++_passed;
        }

        int finish() const {
            std::cout << _name << ": " << _passed << " passed\n";
            return 0;
        }

    private:
        std::string_view _name;
        int _passed{};
};

inline bool near(double left, double right, double tolerance = 1e-9) {
    return std::abs(left - right) <= tolerance;
}

} // namespace actuator_test
