#pragma once
#include <random>
#include <vector>
#include "Parameter.h"
#include "RngEngine.h"

namespace eventide {
    struct Draw {
        double R0, k, r, alpha, theta;
    };


    enum class DrawID : int {
        R0 = 0, k = 1, r = 2, alpha = 3, theta = 4
    };

    /**
     * @brief Latin‐Hypercube sampler over an arbitrary set of Parameters.
     */
    class LatinHypercubeSampler {
    public:
        /**
         * @param params    List of parameters to sample (size d).
         * @param rng       RngEngine for shuffle
         * @param scramble  If true, shuffle strata per dimension.
         */
        explicit LatinHypercubeSampler(std::vector<Parameter> params, const RngEngine& rng, bool scramble = true);

        LatinHypercubeSampler(const LatinHypercubeSampler& other, const RngEngine& rng):
            params_(other.params_), rng_(rng), scramble_(other.scramble_) {}

        /**
         * @brief Draw a block of n samples, each a d‐vector.
         * @return vector of length n, each entry is a vector<double> of length d.
         */
        std::vector<Draw> sampleBlock(int n);

        std::vector<Parameter> parameters() const { return std::vector(params_); }

    private:
        const std::vector<Parameter> params_;
        RngEngine rng_;
        const bool scramble_;

        // Helper: generate a random permutation 0..n-1
        std::vector<int> shuffledIndices(const int n) {
            std::vector<int> idx(n);
            for (int i = 0; i < n; ++i) idx[i] = i;
            if (scramble_) {
                std::shuffle(idx.begin(), idx.end(), std::mt19937(rng_.nextUInt32()));
            }
            return idx;
        }
    };
}
