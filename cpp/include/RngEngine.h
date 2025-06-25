#pragma once
#include <mutex>

namespace eventide {
    /**
     * @brief High-performance RNG engine offering
     *   • Uniform [0,1)
     *   • Gaussian via Ziggurat
     *   • Gamma via Marsaglia–Tsang
     *   • Negative-Binomial via Gamma→Poisson mixing
     */
    class RngEngine {
    public:
        /**
         * @param seed  Optional seed (default = time ^ thread_id).
         */
        explicit RngEngine(uint64_t seed = defaultSeed());

        /** @return a double ∈ [0,1) */
        double uniform();

        /** @brief Draw one Normal(mean, stddev).
         *  @param mean Mean
         *  @param stddev  Must be > 0.
         *  @return a Normal(mean, stddev) variate */
        double normal(double mean, double stddev);

        /** @brief Draw one Gamma(shape, scale) variate.
         *  @param shape Must be > 0.
         *  @param scale Must be > 0.
         *  @return a Gamma(shape, scale) variate */
        double gamma(double shape, double scale);

        /** @brief Draw one NB(r,p) variate via Gamma→Poisson.
         *  @param r  Must be > 0.
         *  @param mu  Must be in (0,1).
         *  @return a NegBinomial(r, p) variate */
        int negBinomial(double r, double mu);

        /** Default seed generator (clock ^ thread_id) */
        static uint64_t defaultSeed();

        /** @return next 32-bit uniform integer via PCG */
        uint32_t nextUInt32();

    private:
        // --- PCG state ---
        uint64_t state_;
        uint64_t increment_;
        bool haveSpare_ = false;
        double spare_ = 0.0;


        // --- Core RNG primitives ---


        /**@return a single sample from StandardNormal(0,1) via Box-Muller */
        float sampleBoxMuller();

        /** Marsaglia–Tsang algorithm for Gamma(shape,1) */
        double sampleGammaShape1(double shape);
    };
}
