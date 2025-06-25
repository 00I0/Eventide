#include "Simulator.h"

#include <cassert>
#include <iostream>

using namespace eventide;

Simulator::Simulator(const LatinHypercubeSampler& sampler,
                     const Scenario& scenario,
                     const CriterionGroup& criteria,
                     const DataCollectorGroup& collectors,
                     const int numTrajectories,
                     const int chunkSize,
                     const int T_run,
                     const int maxCases,
                     const int maxWorkers,
                     const int cutoffDay)
    : sampler_(sampler),
      scenario_(scenario),
      criteria_(criteria),
      collectors_(collectors),
      numTrajectories_(numTrajectories),
      chunkSize_(chunkSize),
      T_run_(T_run),
      maxCases_(maxCases),
      maxWorkers_(maxWorkers),
      cutoffDay_(cutoffDay) {
    assert(numTrajectories_ > 0 && chunkSize_ > 0 && T_run_ >= 0);
}


void Simulator::run() {
    const int nChunks = (numTrajectories_ + chunkSize_ - 1) / chunkSize_;
    std::atomic nextChunk{0};

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
                const int chunk = nextChunk.fetch_add(1, std::memory_order_relaxed);
                if (chunk >= nChunks) break;

                processChunk(chunk, wk.rng, wk.criteria, wk.collectors, wk.scenario, wk.sampler);
            }
        });
    }

    for (auto& wk : workers) wk.thread.join();
    for (auto& wk : workers) collectors_.merge(wk.collectors);
}

void Simulator::processChunk(const int chunkIndex, RngEngine& rng, CriterionGroup& criterionGroup,
                             DataCollectorGroup& collectorGroup, Scenario& scenario,
                             LatinHypercubeSampler& sampler) const {
    const int remain = numTrajectories_ - chunkIndex * chunkSize_;
    const int blockSz = std::min(chunkSize_, remain);
    const auto block = sampler.sampleBlock(blockSz);

    for (const auto& draw : block) {
        if (draw.R0 * draw.r > 10) continue;
        if (draw.alpha * draw.theta < 1 || draw.alpha * draw.theta > 50) continue;

        criterionGroup.reset();
        collectorGroup.reset();
        scenario.reset();
        if (processTrajectory(draw, rng, criterionGroup, collectorGroup, scenario))
            collectorGroup.save();
    }
}


bool Simulator::processTrajectory(const Draw& originalDraw, RngEngine& rng, CriterionGroup& criterionGroup,
                                  DataCollectorGroup& collectorGroup, Scenario& scenario) const {
    // std::cout << "Processing trajectory" << std::endl;
    Draw draw = originalDraw;

    // process the root
    const int nRoot = rng.negBinomial(draw.k, draw.R0);
    int cases = 0;
    if (!criterionGroup.checkRoot(nRoot)) return false;

    std::priority_queue<double, std::vector<double>, std::greater<double>> heap;
    for (int i = 0; i < nRoot; i++) {
        double newInfectionTime = rng.gamma(draw.alpha, draw.theta);
        criterionGroup.registerTime(newInfectionTime);
        if (criterionGroup.earlyReject()) return false;
        heap.push(newInfectionTime);
        collectorGroup.registerTime(newInfectionTime);
        cases++;
    }
    // std::cout << cases << std::endl;

    while (!heap.empty() && cases < maxCases_) {
        // std::cout << "in while " << heap.top() << std::endl;
        const bool accepted = simulateSegment(heap, cases, draw, rng, criterionGroup, collectorGroup,
                                              scenario.nextTime());
        if (!accepted)
            return false;

        if (heap.empty())
            break;

        const double currentTime = heap.top();
        for (double changeTime = scenario.nextTime(); changeTime <= currentTime; changeTime = scenario.nextTime())
            scenario.applyNext(draw, originalDraw);
    }

    collectorGroup.recordDraw(originalDraw);
    return criterionGroup.finalPassed();
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

