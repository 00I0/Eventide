// CompiledExpression.h
#pragma once
#include <string>

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <dlfcn.h>

#include "Sampler.h"

namespace eventide {
    class CompiledExpression {
    public:
        explicit CompiledExpression(const std::string& expr);
        double eval(const Draw& draw) const;

    private:
        struct Impl;

        std::shared_ptr<Impl> impl_;
    };
}
