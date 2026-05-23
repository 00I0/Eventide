#include "AlternatingSimulator.h"

#include <algorithm>
#include <cassert>
#include <cstring>

using namespace eventide;

namespace {
    uint64_t mix_seed(const double a, const double b, const double c, const double d, const double e) {
        uint64_t u1, u2, u3, u4, u5;
        std::memcpy(&u1, &a, 8);
        std::memcpy(&u2, &b, 8);
        std::memcpy(&u3, &c, 8);
        std::memcpy(&u4, &d, 8);
        std::memcpy(&u5, &e, 8);

        constexpr uint64_t P1 = 0xa0761d6472719bbdULL;
        constexpr uint64_t P2 = 0xe7037ed1a0b428dbULL;
        constexpr uint64_t SEED = 0x9E3779B97f4A7C15ULL;

        const uint64_t laneA = (u1 ^ P1) * (u2 ^ P2);
        const uint64_t laneB = (u3 ^ P1) * (u4 ^ P2);
        const uint64_t laneC = (u5 ^ P1) * (SEED ^ P2);

        uint64_t hash = (laneA ^ laneB) + laneC;
        hash ^= hash >> 33;
        hash *= P2;
        hash ^= hash >> 29;
        return hash;
    }

    uint64_t fast_seed_10(const Draw& hostDraw, const Draw& vectorDraw) {
        uint64_t hash = mix_seed(hostDraw.R0, hostDraw.k, hostDraw.r, hostDraw.alpha, hostDraw.theta);
        hash ^= mix_seed(vectorDraw.R0, vectorDraw.k, vectorDraw.r, vectorDraw.alpha, vectorDraw.theta)
                + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        return hash;
    }

    inline Species otherSpecies(const Species species) noexcept {
        return species == Species::HOST ? Species::VECTOR : Species::HOST;
    }

    inline CriterionGroup& criteriaFor(const Species species,
                                       CriterionGroup& hostCriteria,
                                       CriterionGroup& vectorCriteria) noexcept {
        return species == Species::HOST ? hostCriteria : vectorCriteria;
    }

    inline DataCollectorGroup& collectorsFor(const Species species,
                                             DataCollectorGroup& hostCollectors,
                                             DataCollectorGroup& vectorCollectors) noexcept {
        return species == Species::HOST ? hostCollectors : vectorCollectors;
    }

    inline Draw& drawFor(const Species species, Draw& hostDraw, Draw& vectorDraw) noexcept {
        return species == Species::HOST ? hostDraw : vectorDraw;
    }

    inline const Draw& drawFor(const Species species, const Draw& hostDraw, const Draw& vectorDraw) noexcept {
        return species == Species::HOST ? hostDraw : vectorDraw;
    }
}

AlternatingSimulator::AlternatingSimulator(const Sampler& hostSamplerProto,
                                           const Sampler& vectorSamplerProto,
                                           const Scenario& hostScenario,
                                           const Scenario& vectorScenario,
                                           const CriterionGroup& hostCriteria,
                                           const CriterionGroup& vectorCriteria,
                                           const DataCollectorGroup& hostCollectors,
                                           const DataCollectorGroup& vectorCollectors,
                                           const int64_t numTrajectories,
                                           const int64_t minAccepted,
                                           const int chunkSize,
                                           const double T_run,
                                           const int maxCases,
                                           const int maxWorkers,
                                           const CompiledExpression& hostParamValidator,
                                           const CompiledExpression& vectorParamValidator,
                                           const Species rootSpecies)
    : hostSamplerProto_(hostSamplerProto),
      vectorSamplerProto_(vectorSamplerProto),
      hostScenarioProto_(hostScenario),
      vectorScenarioProto_(vectorScenario),
      hostCriteriaProto_(hostCriteria),
      vectorCriteriaProto_(vectorCriteria),
      hostCollectors_(hostCollectors),
      vectorCollectors_(vectorCollectors),
      numTrajectories_(numTrajectories),
      minAccepted_(minAccepted),
      chunkSize_(chunkSize),
      maxCases_(maxCases),
      maxWorkers_(maxWorkers),
      T_run_(T_run),
      hostParamValidator_(hostParamValidator),
      vectorParamValidator_(vectorParamValidator),
      rootSpecies_(rootSpecies) {
    assert(numTrajectories_ > 0 && chunkSize_ > 0 && T_run_ >= 0);
}

void AlternatingSimulator::run() {
    const int64_t nChunks = (numTrajectories_ + chunkSize_ - 1) / chunkSize_;
    std::atomic<int64_t> nextChunk{0};
    numAccepted_.store(0, std::memory_order_release);
    numProcessedTrajectories_.store(0, std::memory_order_release);

    std::vector<RngEngine> rngs(maxWorkers_);
    for (int i = 0; i < maxWorkers_; ++i) rngs[i] = RngEngine(RngEngine::defaultSeed() + i);

    auto hostSamplers = hostSamplerProto_.split(maxWorkers_, rngs);
    auto vectorSamplers = vectorSamplerProto_.split(maxWorkers_, rngs);

    struct WorkerCtx {
        std::unique_ptr<Sampler> hostSampler;
        std::unique_ptr<Sampler> vectorSampler;
        RngEngine rng;
        CriterionGroup hostCriteria;
        CriterionGroup vectorCriteria;
        DataCollectorGroup hostCollectors;
        DataCollectorGroup vectorCollectors;
        Scenario hostScenario;
        Scenario vectorScenario;
        CompiledExpression hostParamValidator;
        CompiledExpression vectorParamValidator;
        std::thread thread;
        EventHeap heap;
        std::vector<Draw> hostDrawBlock;
        std::vector<Draw> vectorDrawBlock;
    };
    std::vector<WorkerCtx> workers;
    workers.reserve(maxWorkers_);

    for (int i = 0; i < maxWorkers_; ++i) {
        workers.push_back({
            std::move(hostSamplers[i]),
            std::move(vectorSamplers[i]),
            rngs[i],
            hostCriteriaProto_,
            vectorCriteriaProto_,
            hostCollectors_,
            vectorCollectors_,
            hostScenarioProto_,
            vectorScenarioProto_,
            hostParamValidator_,
            vectorParamValidator_,
            {},
            {},
            {},
            {}
        });
        auto& wk = workers.back();
        wk.heap.reserve_space(maxCases_);
        wk.hostDrawBlock.reserve(chunkSize_);
        wk.vectorDrawBlock.reserve(chunkSize_);
        wk.thread = std::thread([&wk, &nextChunk, nChunks, this] {
            while (numAccepted_.load(std::memory_order_acquire) < minAccepted_) {
                if (wk.hostSampler->hasFinished() || wk.vectorSampler->hasFinished()) break;
                const int64_t chunkIdx = nextChunk.fetch_add(1, std::memory_order_acq_rel);
                if (chunkIdx >= nChunks) break;

                const int chunkAccepted = processChunk(
                    chunkIdx,
                    wk.rng,
                    wk.hostCriteria, wk.vectorCriteria,
                    wk.hostCollectors, wk.vectorCollectors,
                    wk.hostScenario, wk.vectorScenario,
                    *wk.hostSampler, *wk.vectorSampler,
                    wk.hostParamValidator, wk.vectorParamValidator,
                    wk.heap, wk.hostDrawBlock, wk.vectorDrawBlock
                );

                if (chunkAccepted > 0) numAccepted_.fetch_add(chunkAccepted, std::memory_order_acq_rel);
            }
        });
    }

    for (auto& wk : workers) wk.thread.join();
    for (auto& wk : workers) {
        hostCollectors_.merge(wk.hostCollectors);
        vectorCollectors_.merge(wk.vectorCollectors);
    }
}

int AlternatingSimulator::processChunk(const int64_t chunkIndex,
                                       RngEngine& rng,
                                       CriterionGroup& hostCriteria, CriterionGroup& vectorCriteria,
                                       DataCollectorGroup& hostCollectors, DataCollectorGroup& vectorCollectors,
                                       Scenario& hostScenario, Scenario& vectorScenario,
                                       Sampler& hostSampler, Sampler& vectorSampler,
                                       CompiledExpression& hostParamValidator, CompiledExpression& vectorParamValidator,
                                       EventHeap& heap,
                                       std::vector<Draw>& hostDrawBlock, std::vector<Draw>& vectorDrawBlock) {
    const int64_t remain = numTrajectories_ - chunkIndex * static_cast<int64_t>(chunkSize_);
    if (remain <= 0) return 0;
    const int blockSz = std::min<int64_t>(chunkSize_, remain);

    hostSampler.sampleBlockInto(blockSz, hostDrawBlock);
    vectorSampler.sampleBlockInto(blockSz, vectorDrawBlock);

    const int actualBlockSz = std::min(hostDrawBlock.size(), vectorDrawBlock.size());
    if (actualBlockSz <= 0) return 0;

    int acceptedLocal = 0;
    int processedLocal = 0;
    for (int i = 0; i < actualBlockSz; ++i) {
        const auto& hostDraw = hostDrawBlock[i];
        const auto& vectorDraw = vectorDrawBlock[i];
        processedLocal++;

        if (!hostParamValidator.eval(hostDraw) || !vectorParamValidator.eval(vectorDraw)) {
            hostSampler.reportResult(TrajectoryResult::REJECTED);
            vectorSampler.reportResult(TrajectoryResult::REJECTED);
            continue;
        }

        hostCriteria.reset();
        vectorCriteria.reset();
        hostCollectors.reset();
        vectorCollectors.reset();
        hostScenario.reset();
        vectorScenario.reset();
        heap.clear();

        rng.setSeed(fast_seed_10(hostDraw, vectorDraw));
        const auto trajectoryResult = processTrajectory(
            hostDraw, vectorDraw, rng,
            hostCriteria, vectorCriteria,
            hostCollectors, vectorCollectors,
            hostScenario, vectorScenario,
            heap
        );
        hostSampler.reportResult(trajectoryResult);
        vectorSampler.reportResult(trajectoryResult);
        if (trajectoryResult != TrajectoryResult::REJECTED) {
            hostCollectors.save(trajectoryResult);
            vectorCollectors.save(trajectoryResult);
            acceptedLocal++;
            if (numAccepted_.load(std::memory_order_acquire) + acceptedLocal >= minAccepted_) break;
        }
    }

    numProcessedTrajectories_.fetch_add(processedLocal, std::memory_order_acq_rel);
    return acceptedLocal;
}

TrajectoryResult AlternatingSimulator::processTrajectory(const Draw& originalHostDraw, const Draw& originalVectorDraw,
                                                         RngEngine& rng,
                                                         CriterionGroup& hostCriteria, CriterionGroup& vectorCriteria,
                                                         DataCollectorGroup& hostCollectors,
                                                         DataCollectorGroup& vectorCollectors,
                                                         Scenario& hostScenario, Scenario& vectorScenario,
                                                         EventHeap& heap) const {
    Draw hostDraw = originalHostDraw;
    Draw vectorDraw = originalVectorDraw;

    int cases = 1;
    const Draw& rootDraw = drawFor(rootSpecies_, hostDraw, vectorDraw);
    const Species childSpecies = otherSpecies(rootSpecies_);
    const int nRoot = rng.negBinomial(rootDraw.k, rootDraw.R0);

    if (!criteriaFor(rootSpecies_, hostCriteria, vectorCriteria).checkRoot(nRoot)) return TrajectoryResult::REJECTED;
    collectorsFor(rootSpecies_, hostCollectors, vectorCollectors).registerTime(0.0, 0.0);

    for (int i = 0; i < nRoot; ++i) {
        const double newInfectionTime = rng.gamma(rootDraw.alpha, rootDraw.theta);
        criteriaFor(childSpecies, hostCriteria, vectorCriteria).registerTime(newInfectionTime);
        if (criteriaFor(childSpecies, hostCriteria, vectorCriteria).earlyReject()) return TrajectoryResult::REJECTED;
        heap.push({newInfectionTime, childSpecies});
        collectorsFor(childSpecies, hostCollectors, vectorCollectors).registerTime(0.0, newInfectionTime);
        cases++;
    }

    while (!heap.empty() && cases < maxCases_) {
        const double until = std::min(hostScenario.nextTime(T_run_), vectorScenario.nextTime(T_run_));
        if (!simulateSegment(heap, cases,
                             hostDraw,
                             vectorDraw,
                             rng,
                             hostCriteria, vectorCriteria,
                             hostCollectors, vectorCollectors,
                             until)) {
            return TrajectoryResult::REJECTED;
        }

        if (heap.empty()) break;

        const double currentTime = heap.top().time;
        if (currentTime > T_run_) break;

        while (hostScenario.nextTime() <= currentTime) hostScenario.applyNext(hostDraw, originalHostDraw);
        while (vectorScenario.nextTime() <= currentTime) vectorScenario.applyNext(vectorDraw, originalVectorDraw);
    }

    hostCollectors.recordDraw(originalHostDraw);
    vectorCollectors.recordDraw(originalVectorDraw);
    if (!hostCriteria.finalPassed() || !vectorCriteria.finalPassed()) return TrajectoryResult::REJECTED;

    if (cases >= maxCases_) return TrajectoryResult::CAPPED_AT_MAX_CASES;
    if (!heap.empty() && heap.top().time > T_run_) return TrajectoryResult::CAPPED_AT_T_RUN;
    return TrajectoryResult::ACCEPTED;
}

bool AlternatingSimulator::simulateSegment(EventHeap& heap, int& cases,
                                           Draw& hostDraw,
                                           Draw& vectorDraw,
                                           RngEngine& rng,
                                           CriterionGroup& hostCriteria, CriterionGroup& vectorCriteria,
                                           DataCollectorGroup& hostCollectors, DataCollectorGroup& vectorCollectors,
                                           const double until) const {
    while (!heap.empty() && cases < maxCases_) {
        const Event parentEvent = heap.top();
        if (parentEvent.time > until) break;
        heap.pop();

        Draw& infectorDraw = drawFor(parentEvent.species, hostDraw, vectorDraw);
        const Species newSpecies = otherSpecies(parentEvent.species);
        const int nInfections = rng.negBinomial(infectorDraw.k, infectorDraw.r * infectorDraw.R0);

        CriterionGroup& newSpeciesCriteria = criteriaFor(newSpecies, hostCriteria, vectorCriteria);
        DataCollectorGroup& newSpeciesCollectors = collectorsFor(newSpecies, hostCollectors, vectorCollectors);

        for (int i = 0; i < nInfections; ++i) {
            const double newInfectionTime = rng.gamma(infectorDraw.alpha, infectorDraw.theta) + parentEvent.time;
            newSpeciesCriteria.registerTime(newInfectionTime);
            if (newSpeciesCriteria.earlyReject()) return false;
            heap.push({newInfectionTime, newSpecies});
            newSpeciesCollectors.registerTime(parentEvent.time, newInfectionTime);
            cases++;
        }
    }

    return true;
}
