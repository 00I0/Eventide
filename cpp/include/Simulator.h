#pragma once
#include "Sampler.h"
#include "Scenario.h"
#include "Criterion.h"
#include "Collector.h"
#include "TrajectoryResult.h"

#include <vector>
#include <thread>

namespace eventide {
    /**
     * @brief Orchestrates parameter sampling, parallel branching, criteria, and data collection.
     */


    class Simulator {
    public:
        Simulator(const LatinHypercubeSampler& sampler,
                  const Scenario& scenario,
                  const CriterionGroup& criteria,
                  const DataCollectorGroup& collectors,
                  int64_t numTrajectories,
                  int chunkSize,
                  int T_run,
                  int maxCases,
                  int maxWorkers,
                  int cutoffDay);

        void run();

        const DataCollectorGroup& collectors() const { return collectors_; }

    private:
        // Configuration
        const LatinHypercubeSampler sampler_;
        const Scenario scenario_;
        const CriterionGroup criteria_;
        DataCollectorGroup collectors_;
        const int64_t numTrajectories_;
        const int chunkSize_, T_run_, maxCases_, maxWorkers_, cutoffDay_;

        bool simulateSegment(std::priority_queue<double, std::vector<double>, std::greater<double>>& heap,
                             int& cases, const Draw& draw, RngEngine& rng, CriterionGroup& criterionGroup,
                             DataCollectorGroup& collectorGroup, double until) const;

        TrajectoryResult processTrajectory(const Draw& originalDraw, RngEngine& rng, CriterionGroup& criterionGroup,
                                           DataCollectorGroup& collectorGroup, Scenario& scenario) const;


        void processChunk(int64_t chunkIndex, RngEngine& rng, CriterionGroup& criterionGroup,
                          DataCollectorGroup& collectorGroup, Scenario& scenario, LatinHypercubeSampler& sampler) const;
    };
}
