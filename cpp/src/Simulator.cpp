#include "Simulator.h"

#include <cassert>
#include <iostream>

using namespace eventide;

Simulator::Simulator(const Sampler& samplerProto,
                     const Scenario& scenario,
                     const CriterionGroup& criteria,
                     const DataCollectorGroup& collectors,
                     const int64_t numTrajectories,
                     const int64_t minAccepted,
                     const int chunkSize,
                     const double T_run,
                     const int maxCases,
                     const int maxWorkers,
                     const CompiledExpression& paramValidator)
    : samplerProto_(samplerProto),
      scenario_(scenario),
      criteria_(criteria),
      collectors_(collectors),
      numTrajectories_(numTrajectories),
      minAccepted_(minAccepted),
      chunkSize_(chunkSize),
      maxCases_(maxCases),
      maxWorkers_(maxWorkers),
      T_run_(T_run),
      paramValidator_(paramValidator) {
    assert(numTrajectories_ > 0 && chunkSize_ > 0 && T_run_ >= 0);
}


void Simulator::run() {
    const int64_t nChunks = (numTrajectories_ + chunkSize_ - 1) / chunkSize_;
    std::atomic<int64_t> nextChunk{0};
    numAccepted_.store(0, std::memory_order_release);
    numProcessedTrajectories_ = 0;

    std::vector<RngEngine> rngs(maxWorkers_);
    for (int i = 0; i < maxWorkers_; ++i) rngs[i] = RngEngine(RngEngine::defaultSeed() + i);

    auto samplers = samplerProto_.split(maxWorkers_, rngs);

    struct WorkerCtx {
        std::unique_ptr<Sampler> sampler;
        RngEngine rng;
        CriterionGroup criteria;
        DataCollectorGroup collectors;
        Scenario scenario;
        CompiledExpression paramValidator;
        std::thread thread;
        EventHeap heap;
        std::vector<Draw> drawBlock;
    };
    std::vector<WorkerCtx> workers;
    workers.reserve(maxWorkers_);


    for (int i = 0; i < maxWorkers_; ++i) {
        workers.push_back({
            std::move(samplers[i]), rngs[i], criteria_, collectors_, scenario_, paramValidator_, {}, {}, {}
        });
        auto& wk = workers.back();
        wk.heap.reserve_space(maxCases_);
        wk.drawBlock.reserve(chunkSize_);
        wk.thread = std::thread([&wk, &nextChunk, nChunks, this] {
            while (numAccepted_.load(std::memory_order_acquire) < minAccepted_) {
                const int64_t chunkIdx = nextChunk.fetch_add(1, std::memory_order_acq_rel);
                if (chunkIdx >= nChunks || wk.sampler->hasFinished()) break;
                const int chunkAccepted = processChunk(chunkIdx, wk.rng, wk.criteria, wk.collectors, wk.scenario,
                                                       *wk.sampler, wk.paramValidator, wk.heap, wk.drawBlock);

                if (chunkAccepted > 0) numAccepted_.fetch_add(chunkAccepted, std::memory_order_acq_rel);
            }
        });
    }

    for (auto& wk : workers) wk.thread.join();
    for (auto& wk : workers) collectors_.merge(wk.collectors);
    numProcessedTrajectories_ = std::min(nextChunk.load(std::memory_order_acquire), nChunks)
                                * static_cast<int64_t>(chunkSize_);
}


static uint64_t fast_seed_5(const double d1, const double d2, const double d3, const double d4, const double d5) {
    uint64_t u1, u2, u3, u4, u5;
    std::memcpy(&u1, &d1, 8);
    std::memcpy(&u2, &d2, 8);
    std::memcpy(&u3, &d3, 8);
    std::memcpy(&u4, &d4, 8);
    std::memcpy(&u5, &d5, 8);

    // 2. Constants (WyHash Primes):
    constexpr uint64_t P1 = 0xa0761d6472719bbdULL;
    constexpr uint64_t P2 = 0xe7037ed1a0b428dbULL;
    constexpr uint64_t SEED = 0x9E3779B97f4A7C15ULL; // Golden Ratio


    const uint64_t lane_a = (u1 ^ P1) * (u2 ^ P2);
    const uint64_t lane_b = (u3 ^ P1) * (u4 ^ P2);
    const uint64_t lane_c = (u5 ^ P1) * (SEED ^ P2);

    uint64_t hash = (lane_a ^ lane_b) + lane_c;
    hash ^= hash >> 33;
    hash *= P2;
    hash ^= hash >> 29;

    return hash;
}


int Simulator::processChunk(const int64_t chunkIndex, RngEngine& rng, CriterionGroup& criterionGroup,
                            DataCollectorGroup& collectorGroup, Scenario& scenario, Sampler& sampler,
                            CompiledExpression& paramValidator, EventHeap& heap, std::vector<Draw>& drawBlock) const {
    const int64_t remain = numTrajectories_ - chunkIndex * static_cast<int64_t>(chunkSize_);
    const int blockSz = std::min(static_cast<int64_t>(chunkSize_), remain);
    sampler.sampleBlockInto(blockSz, drawBlock);

    int acceptedLocal = 0;
    for (const auto& draw : drawBlock) {
        if (!paramValidator.eval(draw)) continue;

        criterionGroup.reset();
        collectorGroup.reset();
        scenario.reset();
        heap.clear();

        // rng.setSeed(42);
        rng.setSeed(fast_seed_5(draw.R0, draw.k, draw.r, draw.alpha, draw.theta));
        auto const trajectoryResult = processTrajectory(draw, rng, criterionGroup, collectorGroup, scenario, heap);
        sampler.reportResult(trajectoryResult);
        if (trajectoryResult != TrajectoryResult::REJECTED) {
            collectorGroup.save(trajectoryResult);
            acceptedLocal++;
            if (numAccepted_.load(std::memory_order_acquire) + acceptedLocal >= minAccepted_) break;
        }
    }
    return acceptedLocal;
}


TrajectoryResult Simulator::processTrajectory(const Draw& originalDraw, RngEngine& rng, CriterionGroup& criterionGroup,
                                              DataCollectorGroup& collectorGroup, Scenario& scenario,
                                              EventHeap& heap) const {
    Draw draw = originalDraw;

    // process the root
    const int nRoot = rng.negBinomial(draw.k, draw.R0);
    int cases = 1;
    if (!criterionGroup.checkRoot(nRoot)) return TrajectoryResult::REJECTED;
    collectorGroup.registerTime(0, 0);

    for (int i = 0; i < nRoot; i++) {
        const double newInfectionTime = rng.gamma(draw.alpha, draw.theta);
        criterionGroup.registerTime(newInfectionTime);
        if (criterionGroup.earlyReject()) return TrajectoryResult::REJECTED;
        heap.push(newInfectionTime);
        collectorGroup.registerTime(0, newInfectionTime);
        cases++;
    }

    while (!heap.empty() && cases < maxCases_) {
        const bool accepted = simulateSegment(heap, cases, draw, rng, criterionGroup, collectorGroup,
                                              scenario.nextTime(T_run_));
        if (!accepted)
            return TrajectoryResult::REJECTED;

        if (heap.empty())
            break;

        const double currentTime = heap.top();
        if (heap.top() > T_run_)
            break;

        for (double changeTime = scenario.nextTime(); changeTime <= currentTime; changeTime = scenario.nextTime())
            scenario.applyNext(draw, originalDraw);
    }

    collectorGroup.recordDraw(originalDraw);
    if (!criterionGroup.finalPassed()) return TrajectoryResult::REJECTED;

    if (cases >= maxCases_) return TrajectoryResult::CAPPED_AT_MAX_CASES;
    if (!heap.empty() && heap.top() > T_run_) return TrajectoryResult::CAPPED_AT_T_RUN;
    return TrajectoryResult::ACCEPTED;
}

bool Simulator::simulateSegment(EventHeap& heap, int& cases, const Draw& draw, RngEngine& rng,
                                CriterionGroup& criterionGroup, DataCollectorGroup& collectorGroup,
                                const double until) const {
    while (!heap.empty() && cases < maxCases_) {
        const double parentInfectionTime = heap.top();
        if (parentInfectionTime > until) break;

        const int nInfections = rng.negBinomial(draw.k, draw.r * draw.R0);
        heap.pop();

        for (int i = 0; i < nInfections; i++) {
            const double newInfectionTime = rng.gamma(draw.alpha, draw.theta) + parentInfectionTime;
            criterionGroup.registerTime(newInfectionTime);
            if (criterionGroup.earlyReject()) return false;
            heap.push(newInfectionTime);
            collectorGroup.registerTime(parentInfectionTime, newInfectionTime);
            cases++;
        }
    }

    return true;
}
