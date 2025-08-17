#pragma once
/**
 * @file Sampler.h
 * @brief Abstract and concrete parameter samplers.
 */
#include <random>
#include <vector>
#include "Parameter.h"
#include "RngEngine.h"

namespace eventide {
    enum class TrajectoryResult;

    struct Draw {
        double R0, k, r, alpha, theta;
    };


    enum class DrawID : int {
        R0 = 0, k = 1, r = 2, alpha = 3, theta = 4
    };


    /**
     * @brief Abstract base class for all parameter samplers.
     *
     * Defines a uniform interface so Simulator can fork off N thread-local samplers from a single prototype.
     */
    class Sampler {
    public:
        virtual ~Sampler() = default;

        /**
         * @brief Fork this prototype into `numThreads` independent samplers.
         * @param numThreads  Number of worker threads to spawn.
         * @param seeds       Vector of length `numThreads` supplying each thread's RNG.
         * @return            A vector of `numThreads` unique_ptrs to new Sampler instances.
         */
        virtual std::vector<std::unique_ptr<Sampler>> split(int numThreads, const std::vector<RngEngine>& seeds) const =
        0;

        /**
         * @brief Draw up to n samples in one batch.
         * @param n  Number of samples requested.
         * @return   A vector of Draw (possibly fewer than n if exhausted).
         */
        virtual std::vector<Draw> sampleBlock(int n) = 0;

        /**
         * @brief Inform the sampler whether the last draw was accepted.
         * @param result the TrajectoryResult of the trajectory. 
         */
        virtual void reportResult(TrajectoryResult result) = 0;

        /**
         * @brief Check if the sampler has completed its sampling tasks.
         * @return True, if the sampler has sampled all the parameters, it was supposed to,
         *         false if more samples are available.
         */
        virtual bool hasFinished() const = 0;
    };


    /**
     * @brief Latin-Hypercube sampler over an arbitrary set of Parameters.
     *
     * Samples uniformly within strata per dimension, optionally scrambled.
     */
    class LatinHypercubeSampler final : public Sampler {
    public:
        /**
         * @param params    List of parameters to sample (size d).
         * @param rng       RngEngine for shuffle and uniforms.
         * @param scramble  If true, shuffle strata per dimension.
         */
        explicit LatinHypercubeSampler(const std::vector<Parameter>& params, const RngEngine& rng,
                                       bool scramble = true);

        /**
         * @brief Copy-constructor for creating a new thread-local sampler.
         * @param other  Prototype to copy.
         * @param rng    New RNG for this thread.
         */
        LatinHypercubeSampler(const LatinHypercubeSampler& other, const RngEngine& rng)
            : params_(other.params_), rng_(rng), scramble_(other.scramble_) {}

        std::vector<std::unique_ptr<Sampler>> split(int numThreads, const std::vector<RngEngine>& seeds) const override;

        std::vector<Draw> sampleBlock(int n) override;

        void reportResult(TrajectoryResult) override {}

        bool hasFinished() const override { return false; }

    private:
        std::vector<Parameter> params_;
        RngEngine rng_;
        const bool scramble_;

        /**
         * @brief Generate a random permutation of 0…n-1.
         */
        std::vector<int> shuffledIndices(int n);
    };

    /**
     * @brief A sampler that hands out a fixed list of Draws, each tried up to maxTrials_ times before moving on.
     *
     * Each thread only ever sees the draws in its own segment of the global list.
     * Only supports sampleBlock(1).
     */
    class PreselectedSampler final : public Sampler {
    public:
        /**
         * @brief Construct the root prototype.
         * @param draws      Full list of pre‐chosen Draw structs.
         * @param maxTrials  Maximum retry attempts for each draw on rejection.
         */
        PreselectedSampler(const std::vector<Draw>& draws, int maxTrials);

        /**
         * @brief Fork‐constructor for each thread, restricted to [segBegin, segEnd).
         * @param sharedDraws  Shared pointer to the full draws list.
         * @param maxTrials    Same retry limit.
         * @param segBegin     Inclusive start index into *sharedDraws.
         * @param segEnd       Exclusive end index.
         */
        PreselectedSampler(std::shared_ptr<const std::vector<Draw>> sharedDraws, int maxTrials, size_t segBegin,
                           size_t segEnd);

        std::vector<std::unique_ptr<Sampler>> split(int numThreads, const std::vector<RngEngine>& seeds) const override;
        std::vector<Draw> sampleBlock(int n) override;
        void reportResult(TrajectoryResult result) override;
        bool hasFinished() const override;

    private:
        std::shared_ptr<const std::vector<Draw>> draws_;
        const int maxTrials_;

        /* Segment bounds within draws_: each thread only serves [segBegin_, segEnd_). */
        const size_t segBegin_;
        const size_t segEnd_;

        /* Per-thread, mutable state: */
        size_t currentIdx_; /* absolute index in draws_ */
        int currentTrials_; /* how many times we've retried this draw */
    };
}
