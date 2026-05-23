#pragma once
#include "Sampler.h"
#include "Scenario.h"
#include "Criterion.h"
#include "Collector.h"
#include "TrajectoryResult.h"
#include "DaryHeap.h"

#include <vector>
#include <thread>

namespace eventide {
    /**
     * @brief Orchestrates parameter sampling, parallel branching, criteria, and data collection.
     */
    class Simulator {
    public:
        /**
         * @param samplerProto     Prototype sampler to fork into threads.
         * @param scenario         Scenario model.
         * @param criteria         Group of acceptance/rejection criteria.
         * @param collectors       Group of data collectors.
         * @param numTrajectories  Total trajectories to attempt.
         * @param minAccepted      Minimum number of accepted trajectories to reach before stopping early.
         * @param chunkSize        How many draws per batch.
         * @param T_run            Simulation time horizon.
         * @param maxCases         Max cases per trajectory.
         * @param maxWorkers       Number of threads.
         * @param paramValidator   Expression to filter invalid parameters.
         */
        Simulator(const Sampler& samplerProto,
                  const Scenario& scenario,
                  const CriterionGroup& criteria,
                  const DataCollectorGroup& collectors,
                  int64_t numTrajectories,
                  int64_t minAccepted,
                  int chunkSize,
                  double T_run,
                  int maxCases,
                  int maxWorkers,
                  const CompiledExpression& paramValidator);

        /** @brief Run all simulations, merging results into `collectors()`. */
        void run();

        /** @brief Access the merged collectors after `run()`. */
        const DataCollectorGroup& collectors() const { return collectors_; }
        /** @brief Number of accepted trajectories from the last run. */
        int64_t acceptedCount() const { return numAccepted_.load(); }
        /** @brief Number of processed sampled trajectories from the last run. */
        int64_t processedCount() const { return numProcessedTrajectories_; }

    private:
        // Configuration
        const Sampler& samplerProto_;
        const Scenario scenario_;
        const CriterionGroup criteria_;
        DataCollectorGroup collectors_;
        const int64_t numTrajectories_;
        const int64_t minAccepted_;
        const int chunkSize_, maxCases_, maxWorkers_;
        const double T_run_;
        const CompiledExpression paramValidator_;

        std::atomic<int64_t> numAccepted_{0};
        int64_t numProcessedTrajectories_{0};

        using EventHeap = DaryHeap<>;

        bool
        simulateSegment(EventHeap& heap, int& cases, const Draw& draw, RngEngine& rng, CriterionGroup& criterionGroup,
                        DataCollectorGroup& collectorGroup, double until) const;

        TrajectoryResult processTrajectory(const Draw& originalDraw, RngEngine& rng, CriterionGroup& criterionGroup,
                                           DataCollectorGroup& collectorGroup, Scenario& scenario,
                                           EventHeap& heap) const;


        int processChunk(int64_t chunkIndex, RngEngine& rng, CriterionGroup& criterionGroup,
                         DataCollectorGroup& collectorGroup, Scenario& scenario, Sampler& sampler,
                         CompiledExpression& paramValidator, EventHeap& heap, std::vector<Draw>& drawBlock) const;
    };
}
