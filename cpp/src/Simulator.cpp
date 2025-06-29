#include "Simulator.h"

#include <cassert>
#include <iostream>

using namespace eventide;

Simulator::Simulator(const LatinHypercubeSampler& sampler,
                     const Scenario& scenario,
                     const CriterionGroup& criteria,
                     const DataCollectorGroup& collectors,
                     const int64_t numTrajectories,
                     const int chunkSize,
                     const int T_run,
                     const int maxCases,
                     const int maxWorkers,
                     const CompiledExpression& paramValidator)
    : sampler_(sampler),
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

    struct WorkerCtx {
        DataCollectorGroup collectors;
        CriterionGroup criteria;
        Scenario scenario;
        RngEngine rng;
        LatinHypercubeSampler sampler;
        std::thread thread;

        WorkerCtx(const DataCollectorGroup& protoColl, const CriterionGroup& protoCrit, const Scenario& protoScen,
                  const LatinHypercubeSampler& protoSampler, const int seed):
            collectors(protoColl),
            criteria(protoCrit),
            scenario(protoScen),
            rng(seed),
            sampler(protoSampler, rng) {}
    };

    std::vector<WorkerCtx> workers;
    workers.reserve(maxWorkers_);
    for (int i = 0; i < maxWorkers_; ++i)
        workers.emplace_back(collectors_, criteria_, scenario_, sampler_, RngEngine::defaultSeed() + i);

    for (int w = 0; w < maxWorkers_; ++w) {
        WorkerCtx& wk = workers[w];
        wk.thread = std::thread([&, nChunks] {
            while (true) {
                const int64_t chunk = nextChunk.fetch_add(1, std::memory_order_relaxed);
                if (chunk >= nChunks) break;

                processChunk(chunk, wk.rng, wk.criteria, wk.collectors, wk.scenario, wk.sampler);
            }
        });
    }

    for (auto& wk : workers) wk.thread.join();
    for (auto& wk : workers) collectors_.merge(wk.collectors);
}

void Simulator::processChunk(const int64_t chunkIndex, RngEngine& rng, CriterionGroup& criterionGroup,
                             DataCollectorGroup& collectorGroup, Scenario& scenario,
                             LatinHypercubeSampler& sampler) const {
    const int64_t remain = numTrajectories_ - chunkIndex * static_cast<int64_t>(chunkSize_);
    const int blockSz = std::min(static_cast<int64_t>(chunkSize_), remain);
    const auto block = sampler.sampleBlock(blockSz);


    for (const auto& draw : block) {
        if (!paramValidator_.eval(draw)) continue;

        criterionGroup.reset();
        collectorGroup.reset();
        scenario.reset();
        TrajectoryResult trajectoryResult = processTrajectory(draw, rng, criterionGroup, collectorGroup, scenario);
        if (trajectoryResult != TrajectoryResult::REJECTED)
            collectorGroup.save(trajectoryResult);
    }
}


TrajectoryResult Simulator::processTrajectory(const Draw& originalDraw, RngEngine& rng, CriterionGroup& criterionGroup,
                                              DataCollectorGroup& collectorGroup, Scenario& scenario) const {
    Draw draw = originalDraw;

    // process the root
    const int nRoot = rng.negBinomial(draw.k, draw.R0);
    int cases = 0;
    if (!criterionGroup.checkRoot(nRoot)) return TrajectoryResult::REJECTED;

    std::priority_queue<double, std::vector<double>, std::greater<double>> heap;
    for (int i = 0; i < nRoot; i++) {
        double newInfectionTime = rng.gamma(draw.alpha, draw.theta);
        criterionGroup.registerTime(newInfectionTime);
        if (criterionGroup.earlyReject()) return TrajectoryResult::REJECTED;
        heap.push(newInfectionTime);
        collectorGroup.registerTime(newInfectionTime);
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
        const double infectionTime = heap.top();
        if (infectionTime > until) break;

        const int nInfections = rng.negBinomial(draw.k, draw.r * draw.R0);
        heap.pop();

        for (int i = 0; i < nInfections; i++) {
            double newInfectionTime = rng.gamma(draw.alpha, draw.theta) + infectionTime;
            criterionGroup.registerTime(newInfectionTime);
            if (criterionGroup.earlyReject()) return false;
            heap.push(newInfectionTime);
            collectorGroup.registerTime(newInfectionTime);
            cases++;
        }
    }

    return true;
}

