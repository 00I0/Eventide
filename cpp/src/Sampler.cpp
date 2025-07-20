#include "Sampler.h"
#include <algorithm>
#include <random>
#include <cassert>

using namespace eventide;

LatinHypercubeSampler::LatinHypercubeSampler(std::vector<Parameter> params, const RngEngine& rng, const bool scramble)
    : params_(std::move(params)), rng_(rng), scramble_(scramble) {
    assert(!params_.empty());
}


std::vector<int> LatinHypercubeSampler::shuffledIndices(const int n) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    if (scramble_) {
        std::shuffle(idx.begin(), idx.end(), std::mt19937(rng_.nextUInt32()));
    }
    return idx;
}

std::vector<Draw> LatinHypercubeSampler::sampleBlock(const int n) {
    const auto d = params_.size();
    if (d != 5) throw std::runtime_error("sampleBlock: expected exactly 5 parameters for Draw struct.");

    std::vector<std::vector<int>> perm(d);
    for (int j = 0; j < d; ++j) perm[j] = shuffledIndices(n);

    std::vector<Draw> out;
    out.reserve(n);


    for (int i = 0; i < n; ++i) {
        double values[5];
        for (int j = 0; j < d; ++j) {
            if (params_[j].isFixed()) {
                values[j] = params_[j].min;
                continue;
            }
            const int stratum = perm[j][i];
            const double u = (stratum + rng_.uniform()) / static_cast<double>(n);
            const double lo = params_[j].min;
            const double hi = params_[j].max;
            values[j] = lo + u * (hi - lo);
        }
        out.push_back(Draw{values[0], values[1], values[2], values[3], values[4]});
    }
    return out;
}

