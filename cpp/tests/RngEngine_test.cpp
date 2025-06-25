// RngEngine_test.cpp
#include "gtest/gtest.h"
#include "RngEngine.h"
#include <cmath>
#include <vector>
#include <boost/math/distributions.hpp>

using namespace eventide;

static const size_t N = 1'000'000;

static double ks_critical(size_t n) {
    // approximate Kolmogorov-Smirnov critical value for alpha=0.01
    return 1.63 / std::sqrt(n);
}

TEST(RngEngine, Uniform) {
    RngEngine rng(420);
    double mean0 = 0.5;
    double var0 = 1.0 / 12.0;
    double sigma_mean = std::sqrt(var0 / N);
    std::vector<double> values(N);
    for (size_t i = 0; i < N; ++i) {
        values[i] = rng.uniform();
    }
    double sum = 0, sum2 = 0;
    for (auto& x : values) {
        sum += x;
        sum2 += x * x;
    }
    double mean = sum / N;
    double var = sum2 / N - mean * mean;
    EXPECT_NEAR(mean, mean0, 5*sigma_mean);
    EXPECT_NEAR(var/var0, 1.0, 0.10);
    // KS test
    std::sort(values.begin(), values.end());
    double d = 0;
    for (size_t i = 0; i < N; ++i) {
        double F_emp = double(i + 1) / N;
        double F_theo = values[i];
        d = std::max(d, std::abs(F_emp - F_theo));
    }
    EXPECT_LT(d, ks_critical(N));
}

TEST(RngEngine, Normal) {
    RngEngine rng(420);
    for (double mu : {-5, -3, -1, 0, 1, 3, 5, 7, 9, 11}) {
        for (double sigma : {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}) {
            std::vector<double> values;
            values.reserve(N);
            for (size_t i = 0; i < N; ++i) {
                values.push_back(rng.normal(mu, sigma));
            }
            double sum = 0, sum2 = 0;
            for (auto& x : values) {
                sum += x;
                sum2 += x * x;
            }
            double mean0 = mu;
            double var0 = sigma * sigma;
            double mean = sum / N;
            double var = sum2 / N - mean * mean;
            double sigma_mean = std::sqrt(var0 / N);
            EXPECT_NEAR(mean, mean0, 5*sigma_mean) << " μ=" << mu << " σ=" << sigma;
            EXPECT_NEAR(var/var0, 1.0, 0.10) << " μ=" << mu << " σ=" << sigma;
            // KS test
            std::sort(values.begin(), values.end());
            // transform to standard normal CDF
            double d = 0;
            for (size_t i = 0; i < N; ++i) {
                double F_emp = double(i + 1) / N;
                double z = (values[i] - mu) / sigma;
                double F_theo = 0.5 * (1 + std::erf(z / std::sqrt(2)));
                d = std::max(d, std::abs(F_emp - F_theo));
            }
            EXPECT_LT(d, ks_critical(N)) << "Normal KS failed at μ=" << mu << " σ=" << sigma;
        }
    }
}

TEST(RngEngine, Gamma) {
    RngEngine rng(420);
    for (double alpha : {0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 50.0}) {
        for (double theta : {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}) {
            std::vector<double> values;
            values.reserve(N);
            for (size_t i = 0; i < N; ++i) {
                values.push_back(rng.gamma(alpha, theta));
            }
            double sum = 0, sum2 = 0;
            for (auto& x : values) {
                sum += x;
                sum2 += x * x;
            }
            double mean0 = alpha * theta;
            double var0 = alpha * theta * theta;
            double mean = sum / N;
            double var = sum2 / N - mean * mean;
            double sigma_mean = std::sqrt(var0 / N);
            EXPECT_NEAR(mean, mean0, 5*sigma_mean) << " α=" << alpha << " θ=" << theta;
            EXPECT_NEAR(var/var0, 1.0, 0.15) << " α=" << alpha << " θ=" << theta;
            // KS test
            std::sort(values.begin(), values.end());
            double d = 0;
            for (size_t i = 0; i < N; ++i) {
                double F_emp = double(i + 1) / N;
                double F_theo = boost::math::gamma_p(alpha, values[i] / theta);
                d = std::max(d, std::abs(F_emp - F_theo));
            }
            EXPECT_LT(d, ks_critical(N)) << "Gamma KS failed at α=" << alpha << " θ=" << theta;
        }
    }
}

TEST(RngEngine, NegativeBinomial_MeanParam) {
    RngEngine rng(420);

    for (int r : {1, 2, 5, 10, 20})
        for (double mu : {1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0}) {
            // 1) Generate data with (r, μ)
            std::vector<int> data;
            data.reserve(N);
            for (size_t i = 0; i < N; ++i)
                data.push_back(rng.negBinomial(r, mu));

            // 2) Count observed frequencies
            std::map<int, int> freq;
            for (auto x : data) ++freq[x];

            // 3) Build (observed, expected) bins, merging low‐expected tail
            std::vector<double> obs, expct;
            double tailObs = 0.0, tailExp = 0.0;

            // convert mean μ to success‐probability p = r/(r + μ)
            double p = static_cast<double>(r) / (r + mu);
            boost::math::negative_binomial_distribution<> dist(r, p);

            for (auto [k, count] : freq) {
                double pk = boost::math::pdf(dist, k);
                double ek = pk * N;
                if (ek < 5.0) {
                    tailObs += count;
                    tailExp += ek;
                }
                else {
                    obs.push_back(count);
                    expct.push_back(ek);
                }
            }
            if (tailExp > 0.0) {
                obs.push_back(tailObs);
                expct.push_back(tailExp);
            }

            ASSERT_GT(obs.size(), 1u)
                << "Not enough bins for chi2 at r=" << r << " mu=" << mu;

            // 4) Compute chi²
            double chi2 = 0.0;
            size_t bins = obs.size();
            for (size_t i = 0; i < bins; ++i) {
                double o = obs[i], e = expct[i];
                chi2 += (o - e) * (o - e) / e;
            }

            // 5) Compare to χ²_{bins−1}(0.99)
            boost::math::chi_squared chi2dist(double(bins - 1));
            double crit = boost::math::quantile(chi2dist, 0.99);
            EXPECT_LT(chi2, crit)
                << "Chi² GOF failed for NegBin(r=" << r << ", mu=" << mu
                << "): chi2=" << chi2 << ", crit=" << crit << ", bins=" << bins;
        }
}
