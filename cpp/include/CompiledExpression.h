#pragma once
/**
 * @file CompiledExpression.h
 * @brief An arithmetic expression evaluator.
 */
#include <string>
#include "Sampler.h"   // for eventide::Draw

namespace eventide {
    /**
     * @brief Holds a compiled expression for fast repeated evaluation.
     */
    class CompiledExpression {
    public:
        /**
         * @brief Compile a new expression from source.
         * @param expr  arithmetic (and/or boolean) expression over R0, k, r, alpha, theta
         */
        explicit CompiledExpression(const std::string& expr);


        ~CompiledExpression() = default;

        /**
         * @brief Evaluate on one Draw.
         * @param d  the parameter draw
         * @return   the result as double (0=false, nonzero=true)
         */
        double eval(const Draw& d) const;

        /** @brief Get the original source string. */
        std::string expr() const { return expr_; }

    private:
        const std::string expr_;
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };
}
