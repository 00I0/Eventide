#include "Sampler.h"
#include <algorithm>
#include <random>

#include "TrajectoryResult.h"

using namespace eventide;

namespace {
    DrawID nameToDrawID(const std::string& name) {
        std::string n = name;
        std::transform(n.begin(), n.end(), n.begin(), [](const unsigned char c) { return std::tolower(c); });
        if (n == "r0") return DrawID::R0;
        if (n == "k") return DrawID::k;
        if (n == "r") return DrawID::r;
        if (n == "alpha") return DrawID::alpha;
        if (n == "theta") return DrawID::theta;
        throw std::invalid_argument("LatinHypercubeSampler: unknown parameter name '" + name + "'");
    }
}

LatinHypercubeSampler::LatinHypercubeSampler(const std::vector<Parameter>& params, const RngEngine& rng,
                                             const bool scramble)
    : rng_(rng), scramble_(scramble) {
    constexpr size_t D = 5;
    if (params.size() != D)
        throw std::invalid_argument(
            "LatinHypercubeSampler: expected exactly 5 parameters, got " + std::to_string(params.size())
        );


    std::array<std::optional<Parameter>, D> slot{};

    // Populate slots
    for (auto const& p : params) {
        const size_t idx = static_cast<size_t>(nameToDrawID(p.name));
        if (slot[idx])
            throw std::invalid_argument("LatinHypercubeSampler: duplicate parameter for slot " + std::to_string(idx));

        slot[idx].emplace(p);
    }

    params_.reserve(D);
    for (size_t i = 0; i < D; ++i) {
        if (!slot[i])
            throw std::invalid_argument("LatinHypercubeSampler: missing parameter for slot " + std::to_string(i));

        params_.push_back(std::move(*slot[i]));
    }
}


std::vector<std::unique_ptr<Sampler>> LatinHypercubeSampler::split(const int numThreads,
                                                                   const std::vector<RngEngine>& seeds) const {
    std::vector<std::unique_ptr<Sampler>> threads;
    threads.reserve(numThreads);
    for (int i = 0; i < numThreads; i++) {
        threads.push_back(std::make_unique<LatinHypercubeSampler>(*this, seeds[i]));
    }
    return threads;
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


PreselectedSampler::PreselectedSampler(const std::vector<Draw>& draws, const int maxTrials)
    : draws_(std::make_shared<const std::vector<Draw>>(draws)),
      maxTrials_(maxTrials),
      segBegin_(0),
      segEnd_(draws_->size()),
      currentIdx_(0),
      currentTrials_(0) {}


PreselectedSampler::PreselectedSampler(std::shared_ptr<const std::vector<Draw>> sharedDraws, const int maxTrials,
                                       const size_t segBegin, const size_t segEnd)
    : draws_(std::move(sharedDraws)),
      maxTrials_(maxTrials),
      segBegin_(segBegin),
      segEnd_(segEnd),
      currentIdx_(segBegin_),
      currentTrials_(0) {
    if (segBegin_ > segEnd_ || segEnd_ > draws_->size())
        throw std::out_of_range("PreselectedSampler: invalid segment bounds");
}

std::vector<std::unique_ptr<Sampler>> PreselectedSampler::split(const int numThreads,
                                                                const std::vector<RngEngine>& seeds) const {
    const size_t total = draws_->size();
    std::vector<std::unique_ptr<Sampler>> threads;
    threads.reserve(numThreads);


    for (int i = 0; i < numThreads; i++) {
        size_t begin = total * i / numThreads;
        size_t end = total * (i + 1) / numThreads;
        threads.push_back(std::make_unique<PreselectedSampler>(draws_, maxTrials_, begin, end));
    }
    return threads;
}

std::vector<Draw> PreselectedSampler::sampleBlock(const int n) {
    if (n != 1) throw std::runtime_error("PreselectedSampler: only sampleBlock(1) supported");

    std::vector<Draw> out;
    if (currentIdx_ >= segEnd_) return out;

    out.push_back((*draws_)[currentIdx_]);
    return out;
}

void PreselectedSampler::reportResult(const TrajectoryResult result) {
    if (currentIdx_ >= segEnd_) return;

    if (result == TrajectoryResult::REJECTED) currentTrials_++;

    if (currentTrials_ >= maxTrials_ || result != TrajectoryResult::REJECTED) {
        currentIdx_++;
        currentTrials_ = 0;
    }
}

bool PreselectedSampler::hasFinished() const {
    return currentIdx_ >= segEnd_;
}




