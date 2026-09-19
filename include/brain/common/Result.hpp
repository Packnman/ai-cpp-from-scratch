#pragma once

#include "brain/common/Error.hpp"

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace ai::brain {

template <class T> class Result {
        static_assert(!std::is_same_v<T, BrainError>);

    public:
        static Result success(T value) {
            return Result(std::in_place_index<0>, std::move(value));
        }
        static Result failure(BrainError error) {
            return Result(std::in_place_index<1>, std::move(error));
        }

        bool has_value() const noexcept { return _value.index() == 0; }
        explicit operator bool() const noexcept { return has_value(); }

        T &value() {
            if (!has_value())
                throw std::logic_error("Result does not contain a value");
            return std::get<0>(_value);
        }
        const T &value() const {
            if (!has_value())
                throw std::logic_error("Result does not contain a value");
            return std::get<0>(_value);
        }
        BrainError &error() {
            if (has_value())
                throw std::logic_error("Result does not contain an error");
            return std::get<1>(_value);
        }
        const BrainError &error() const {
            if (has_value())
                throw std::logic_error("Result does not contain an error");
            return std::get<1>(_value);
        }

    private:
        template <std::size_t Index, class Value>
        Result(std::in_place_index_t<Index> index, Value &&value)
            : _value(index, std::forward<Value>(value)) {}

        std::variant<T, BrainError> _value;
};

template <> class Result<void> {
    public:
        static Result success() { return Result(); }
        static Result failure(BrainError error) {
            return Result(std::move(error));
        }

        bool has_value() const noexcept { return !_error.has_value(); }
        explicit operator bool() const noexcept { return has_value(); }
        void value() const {
            if (!has_value())
                throw std::logic_error("Result does not contain a value");
        }
        BrainError &error() {
            if (has_value())
                throw std::logic_error("Result does not contain an error");
            return *_error;
        }
        const BrainError &error() const {
            if (has_value())
                throw std::logic_error("Result does not contain an error");
            return *_error;
        }

    private:
        Result() = default;
        explicit Result(BrainError error) : _error(std::move(error)) {}
        std::optional<BrainError> _error;
};

} // namespace ai::brain
