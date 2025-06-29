#pragma once
#include <string>
#include "Sampler.h"   // for eventide::Draw

namespace eventide {
    class CompiledExpression {
    public:
        explicit CompiledExpression(const std::string& expr);


        ~CompiledExpression();

        double eval(const Draw& d) const;

        std::string expr() const { return expr_; };

    private:
        const std::string expr_;
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };
}
