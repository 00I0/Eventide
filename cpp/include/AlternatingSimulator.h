#pragma once

#include "Sampler.h"
#include "Scenario.h"
#include "Criterion.h"
#include "Collector.h"
#include "TrajectoryResult.h"
#include "DaryHeap.h"

#include <atomic>
#include <thread>
#include <vector>

namespace eventide {
    enum class Species : unsigned char { HOST = 0, VECTOR = 1 };

    class AlternatingSimulator {
    public:
        AlternatingSimulator(const Sampler& hostSamplerProto,
                             const Sampler& vectorSamplerProto,
                             const Scenario& hostScenario,
                             const Scenario& vectorScenario,
                             const CriterionGroup& hostCriteria,
                             const CriterionGroup& vectorCriteria,
                             const DataCollectorGroup& hostCollectors,
                             const DataCollectorGroup& vectorCollectors,
                             int64_t numTrajectories,
                             int64_t minAccepted,
                             int chunkSize,
                             double T_run,
                             int maxCases,
                             int maxWorkers,
                             const CompiledExpression& hostParamValidator,
                             const CompiledExpression& vectorParamValidator,
                             Species rootSpecies);

        void run();

        const DataCollectorGroup& hostCollectors() const { return hostCollectors_; }
        const DataCollectorGroup& vectorCollectors() const { return vectorCollectors_; }
        int64_t acceptedCount() const { return numAccepted_.load(); }
        int64_t processedCount() const { return numProcessedTrajectories_.load(); }

    private:
        struct Event {
            double time;
            Species species;

            bool operator<(const Event& other) const noexcept { return time < other.time; }
            bool operator>(const Event& other) const noexcept { return time > other.time; }
        };

        using EventHeap = DaryHeap<Event, 2>;

        const Sampler& hostSamplerProto_;
        const Sampler& vectorSamplerProto_;
        const Scenario hostScenarioProto_;
        const Scenario vectorScenarioProto_;
        const CriterionGroup hostCriteriaProto_;
        const CriterionGroup vectorCriteriaProto_;
        DataCollectorGroup hostCollectors_;
        DataCollectorGroup vectorCollectors_;
        const int64_t numTrajectories_;
        const int64_t minAccepted_;
        const int chunkSize_;
        const int maxCases_;
        const int maxWorkers_;
        const double T_run_;
        const CompiledExpression hostParamValidator_;
        const CompiledExpression vectorParamValidator_;
        const Species rootSpecies_;

        std::atomic<int64_t> numAccepted_{0};
        std::atomic<int64_t> numProcessedTrajectories_{0};

        bool simulateSegment(EventHeap& heap, int& cases,
                             Draw& hostDraw,
                             Draw& vectorDraw,
                             RngEngine& rng,
                             CriterionGroup& hostCriteria, CriterionGroup& vectorCriteria,
                             DataCollectorGroup& hostCollectors, DataCollectorGroup& vectorCollectors,
                             double until) const;

        TrajectoryResult processTrajectory(const Draw& originalHostDraw, const Draw& originalVectorDraw,
                                           RngEngine& rng,
                                           CriterionGroup& hostCriteria, CriterionGroup& vectorCriteria,
                                           DataCollectorGroup& hostCollectors, DataCollectorGroup& vectorCollectors,
                                           Scenario& hostScenario, Scenario& vectorScenario,
                                           EventHeap& heap) const;

        int processChunk(int64_t chunkIndex,
                         RngEngine& rng,
                         CriterionGroup& hostCriteria, CriterionGroup& vectorCriteria,
                         DataCollectorGroup& hostCollectors, DataCollectorGroup& vectorCollectors,
                         Scenario& hostScenario, Scenario& vectorScenario,
                         Sampler& hostSampler, Sampler& vectorSampler,
                         CompiledExpression& hostParamValidator, CompiledExpression& vectorParamValidator,
                         EventHeap& heap,
                         std::vector<Draw>& hostDrawBlock, std::vector<Draw>& vectorDrawBlock);
    };
}
