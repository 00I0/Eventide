#pragma once

namespace eventide {
    /**
     * @brief Describes one model parameter by name and [min,max] range.
     */

    // R0: 0, r: 1, k: 2, alpha: 3, theta: 4.
    struct Parameter {
        const std::string name;
        const double min, max;

        Parameter(const std::string& name_, const double min_, const double max_): name(name_), min(min_), max(max_) {
            if (min > max) throw std::invalid_argument("Parameter: min must be < max for " + name);
        }

        bool isFixed() const { return min == max; }
    };
}
