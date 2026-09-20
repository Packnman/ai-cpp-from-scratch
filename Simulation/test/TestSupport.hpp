#pragma once

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string_view>

namespace simulation_test {

class Suite {
    public:
        explicit Suite(std::string_view name) : _name(name) {}

        void expect(bool condition, std::string_view id,
                    std::string_view description) {
            if (!condition) {
                std::cerr << "FAIL " << id << ": " << description << '\n';
                std::exit(EXIT_FAILURE);
            }
            ++_passed;
        }

        int finish() const {
            std::cout << _name << ": " << _passed << " passed\n";
            return EXIT_SUCCESS;
        }

    private:
        std::string_view _name;
        int _passed{};
};

inline bool near(double left, double right, double tolerance = 1e-9) {
    return std::abs(left - right) <= tolerance;
}

inline std::filesystem::path model(std::string_view relative) {
    return std::filesystem::path{AI_CPP_SIMULATION_MODEL_DIR} / relative;
}

} // namespace simulation_test
