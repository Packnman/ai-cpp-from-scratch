#pragma once

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace brain_test {

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

        void skip(std::string_view id, std::string_view reason) {
            ++_skipped;
            std::cout << "SKIP " << id << ": " << reason << '\n';
        }

        int finish() const {
            std::cout << _name << ": " << _passed << " passed, " << _skipped
                      << " skipped\n";
            return 0;
        }

    private:
        std::string_view _name;
        int _passed{};
        int _skipped{};
};

} // namespace brain_test
