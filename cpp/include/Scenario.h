#pragma once
#include <vector>

#include "Sampler.h"


namespace eventide {
    /**
     * @brief At time `t`, either set one of the 5 parameters to `newValue`,
     *        or restore it to the *original* drawn value.
     */
    struct ParameterChangePoint {
        double time;
        DrawID which;
        bool isRestore;
        double newValue;

        ParameterChangePoint(const double time_, const DrawID which_, const double newValue_)
            : time(time_), which(which_), isRestore(false), newValue(newValue_) {}

        /** ctor for “restore to original” */
        ParameterChangePoint(const double time_, const DrawID which_)
            : time(time_), which(which_), isRestore(true), newValue(0.0) {}
    };

    /**
     * @brief Holds sorted ParameterChangePoints; can apply them in chronological order.
     */
    class Scenario {
    public:
        explicit Scenario(std::vector<ParameterChangePoint> cps);

        Scenario(const Scenario& scenario): cps_(scenario.cps_) {}

        /// Reset to before any changes are applied.
        void reset();

        double nextTime() const {
            return nextIdx_ < cps_.size() ? cps_[nextIdx_].time : std::numeric_limits<double>::infinity();
        }

        /// Apply the next change to the `current` map of parameter values.
        /**
         * @brief Mutate `current[param]`:
         *        • if isRestore:  current[param] = original[param]
         *        • else:          current[param] = cp.newValue
         *
         * @param current   Map of current parameter values
         * @param original  Map of the very first, base-draw values
         */
        void applyNext(Draw& current, const Draw& original);

    private:
        std::vector<ParameterChangePoint> cps_;
        size_t nextIdx_ = 0;
    };
}
