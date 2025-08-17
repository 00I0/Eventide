#include "CompiledExpression.h"
#include <exprtk.hpp>
#include <stdexcept>
#include <memory>

using namespace eventide;


struct CompiledExpression::Impl {
    exprtk::symbol_table<double> symbols;
    exprtk::expression<double> expression;
    exprtk::parser<double> parser;

    double R0, k, r, alpha, theta;

    explicit Impl(const std::string& expr) {
        symbols.add_variable("R0", R0);
        symbols.add_variable("k", k);
        symbols.add_variable("r", r);
        symbols.add_variable("alpha", alpha);
        symbols.add_variable("theta", theta);
        symbols.add_constants(); // math constants (pi, e, etc.)
        expression.register_symbol_table(symbols);


        if (!parser.compile(expr, expression))
            throw std::runtime_error("ExprTk compile error: " + parser.error());
    }
};

CompiledExpression::CompiledExpression(const std::string& expr):
    expr_(expr), impl_(std::make_shared<Impl>(std::move(expr))) {}


double CompiledExpression::eval(const Draw& d) const {
    auto& impl = *impl_;
    impl.R0 = d.R0;
    impl.k = d.k;
    impl.r = d.r;
    impl.alpha = d.alpha;
    impl.theta = d.theta;

    return impl.expression.value();
}

