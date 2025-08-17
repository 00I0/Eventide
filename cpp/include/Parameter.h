#pragma once
/**
 * @file Parameter.h
 * @brief Describes one model parameter by name and [min,max] range.
 */
#include <string>
#include <stdexcept>

namespace eventide {
    /**
     * @brief One dimension in the sampler, fixed or ranged.
     *
     * Valid names: "R0","k","r","alpha","theta".
     */
    struct Parameter {
        const std::string name; /**< parameter name (must map to DrawID) */
        const double min; /**< lower bound */
        const double max; /**< upper bound */

        /**
         * @brief Construct one parameter.
         * @param name_  must match a DrawID name
         * @param min_   minimum permitted value
         * @param max_   maximum permitted value
         * @throws std::invalid_argument if min_ > max_
         */
        Parameter(const std::string& name_, const double min_, const double max_): name(name_), min(min_), max(max_) {
            if (min > max) throw std::invalid_argument("Parameter: min must be < max for " + name);
        }

        /** @brief Is this parameter “fixed” (min == max)? */
        bool isFixed() const noexcept { return min == max; }
    };
}
