#include "Simulator.h"

#include <cassert>
#include <iostream>

using namespace eventide;

Simulator::Simulator(const Sampler& samplerProto,
                     const Scenario& scenario,
                     const CriterionGroup& criteria,
                     const DataCollectorGroup& collectors,
                     const int64_t numTrajectories,
                     const int chunkSize,
                     const int T_run,
                     const int maxCases,
                     const int maxWorkers,
                     const CompiledExpression& paramValidator)
    : samplerProto_(samplerProto),
      scenario_(scenario),
      criteria_(criteria),
      collectors_(collectors),
      numTrajectories_(numTrajectories),
      chunkSize_(chunkSize),
      T_run_(T_run),
      maxCases_(maxCases),
      maxWorkers_(maxWorkers),
      paramValidator_(paramValidator) {
    assert(numTrajectories_ > 0 && chunkSize_ > 0 && T_run_ >= 0);
}


void Simulator::run() {
    const int64_t nChunks = (numTrajectories_ + chunkSize_ - 1) / chunkSize_;
    std::atomic<int64_t> nextChunk{0};

    std::vector<RngEngine> rngs(maxWorkers_);
    for (int i = 0; i < maxWorkers_; ++i) rngs[i] = RngEngine(RngEngine::defaultSeed() + i);

    auto samplers = samplerProto_.split(maxWorkers_, rngs);

    struct WorkerCtx {
        std::unique_ptr<Sampler> sampler;
        RngEngine rng;
        CriterionGroup criteria;
        DataCollectorGroup collectors;
        Scenario scenario;
        std::thread thread;
    };
    std::vector<WorkerCtx> workers;
    workers.reserve(maxWorkers_);


    for (int i = 0; i < maxWorkers_; ++i) {
        workers.push_back({
            std::move(samplers[i]), rngs[i], criteria_, collectors_, scenario_, {}
        });
        auto& wk = workers.back();
        wk.thread = std::thread([&wk, &nextChunk, nChunks, this] {
            while (true) {
                const int64_t chunkIdx = nextChunk.fetch_add(1, std::memory_order_relaxed);
                if (chunkIdx >= nChunks || wk.sampler->hasFinished()) break;
                processChunk(chunkIdx, wk.rng, wk.criteria, wk.collectors, wk.scenario, *wk.sampler);
            }
        });
    }

    for (auto& wk : workers) wk.thread.join();
    for (auto& wk : workers) collectors_.merge(wk.collectors);
}

void Simulator::processChunk(const int64_t chunkIndex, RngEngine& rng, CriterionGroup& criterionGroup,
                             DataCollectorGroup& collectorGroup, Scenario& scenario, Sampler& sampler) const {
    const int64_t remain = numTrajectories_ - chunkIndex * static_cast<int64_t>(chunkSize_);
    const int blockSz = std::min(static_cast<int64_t>(chunkSize_), remain);
    const auto block = sampler.sampleBlock(blockSz);


    for (const auto& draw : block) {
        if (!paramValidator_.eval(draw)) continue;

        criterionGroup.reset();
        collectorGroup.reset();
        scenario.reset();
        auto const trajectoryResult = processTrajectory(draw, rng, criterionGroup, collectorGroup, scenario);
        sampler.reportResult(trajectoryResult);
        if (trajectoryResult != TrajectoryResult::REJECTED)
            collectorGroup.save(trajectoryResult);
    }
}


TrajectoryResult Simulator::processTrajectory(const Draw& originalDraw, RngEngine& rng, CriterionGroup& criterionGroup,
                                              DataCollectorGroup& collectorGroup, Scenario& scenario) const {
    Draw draw = originalDraw;

    // process the root
    const int nRoot = rng.negBinomial(draw.k, draw.R0);
    int cases = 1;
    if (!criterionGroup.checkRoot(nRoot)) return TrajectoryResult::REJECTED;
    collectorGroup.registerTime(0, 0);

    std::priority_queue<double, std::vector<double>, std::greater<double>> heap;
    for (int i = 0; i < nRoot; i++) {
        double newInfectionTime = rng.gamma(draw.alpha, draw.theta);
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
    if (!heap.empty() && heap.top() == std::numeric_limits<double>::infinity()) return TrajectoryResult::REJECTED;

    if (cases >= maxCases_) return TrajectoryResult::CAPPED_AT_MAX_CASES;
    if (!heap.empty() && heap.top() > T_run_) return TrajectoryResult::CAPPED_AT_T_RUN;
    return TrajectoryResult::ACCEPTED;
}

bool Simulator::simulateSegment(std::priority_queue<double, std::vector<double>, std::greater<double>>& heap,
                                int& cases, const Draw& draw, RngEngine& rng, CriterionGroup& criterionGroup,
                                DataCollectorGroup& collectorGroup, const double until) const {
    while (!heap.empty() && cases < maxCases_) {
        const double parentInfectionTime = heap.top();
        if (parentInfectionTime > until) break;

        const int nInfections = rng.negBinomial(draw.k, draw.r * draw.R0);
        heap.pop();

        for (int i = 0; i < nInfections; i++) {
            double newInfectionTime = rng.gamma(draw.alpha, draw.theta) + parentInfectionTime;
            criterionGroup.registerTime(newInfectionTime);
            if (criterionGroup.earlyReject()) return false;
            heap.push(newInfectionTime);
            collectorGroup.registerTime(parentInfectionTime, newInfectionTime);
            cases++;
        }
    }

    return true;
}
