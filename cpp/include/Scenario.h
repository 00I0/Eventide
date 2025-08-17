#pragma once
/**
 * @file Scenario.h
 * @brief Time‑based changes to parameter values during a trajectory.
 */
#include <vector>

#include "CompiledExpression.h"
#include "Sampler.h"


namespace eventide {
    /**
     * @brief At time t, change or restore one parameter.
     */
    struct ParameterChangePoint {
        double time; /**< when (days since start) */
        DrawID which; /**< which parameter to change */
        bool isRestore; /**< true => restores the original draw */
        std::shared_ptr<CompiledExpression> expression; /**< new‐value expression (nullptr => restore) */

        /**
         * @brief Change to expression at time.
         * @param time_         days since start
         * @param which_        which DrawID
         * @param expression_   new expression
         */
        ParameterChangePoint(double time_, DrawID which_, const CompiledExpression& expression_);

        /**
         * @brief Restore original draw at time.
         * @param time_      days since start
         * @param which_     which DrawID
         */
        ParameterChangePoint(double time_, DrawID which_);
    };

    /**
     * @brief Applies a sequence of ParameterChangePoints in time order.
     */
    class Scenario {
    public:
        explicit Scenario(std::vector<ParameterChangePoint> cps);
        Scenario(const Scenario& scenario) = default;

        /** @brief Reset to start of timeline. */
        void reset() noexcept;

        /**
         * @brief Get next change time, capped at infinity.
         * @param cap maximum time
         * @return    next change ≤ cap or inf
         */
        double nextTime(double cap) const noexcept;


        /** @brief Get the next change time (no cap). */
        double nextTime() const noexcept;


        /**
         * @brief Apply the next change to the current parameter set.
         * @param current   mutable draw to update
         * @param original  the original draw values
         */
        void applyNext(Draw& current, const Draw& original);

    private:
        static constexpr double DOUBLE_INF = std::numeric_limits<double>::infinity();
        std::vector<ParameterChangePoint> cps_;
        size_t nextIdx_ = 0;
    };
}
