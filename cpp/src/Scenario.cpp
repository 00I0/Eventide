#include "Scenario.h"
#include <stdexcept>

#include "Sampler.h"

using namespace eventide;

Scenario::Scenario(std::vector<ParameterChangePoint> cps): cps_(std::move(cps)) {
    std::sort(cps_.begin(), cps_.end(), [](auto const& a, auto const& b) { return a.time < b.time; });
}

void Scenario::reset() { nextIdx_ = 0; }


void Scenario::applyNext(Draw& current, const Draw& original) {
    if (nextIdx_ >= cps_.size()) throw std::out_of_range("Scenario::applyNext: no more change points");


    const auto& cp = cps_[nextIdx_];
    const auto which = cp.which;
    const auto isRestore = cp.isRestore;
    const auto newValue = cp.newValue;


    const
        double val = isRestore
                         ? [&] {
                             switch (which) {
                             case DrawID::R0: return original.R0;
                             case DrawID::r: return original.r;
                             case DrawID::k: return original.k;
                             case DrawID::alpha: return original.alpha;
                             case DrawID::theta: return original.theta;
                             }
                             throw std::runtime_error("Scenario::applyNext: unknown draw type");
                         }()
                         : newValue;

    switch (which) {
    case DrawID::R0:
        current.R0 = val;
        break;
    case DrawID::k:
        current.k = val;
        break;
    case DrawID::r:
        current.r = val;
        break;
    case DrawID::alpha:
        current.alpha = val;
        break;
    case DrawID::theta:
        current.theta = val;
        break;
    }

    nextIdx_++;
}
