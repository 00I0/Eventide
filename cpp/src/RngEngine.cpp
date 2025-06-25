#include "RngEngine.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>


using namespace eventide;


//------------------------------------------------------------------------------
// defaultSeed(): mix high-res clock and thread ID for initial seeding
//------------------------------------------------------------------------------
uint64_t RngEngine::defaultSeed() {
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    // return now ^ 0xDEADBEEFCAFEBABEULL;
    const auto tid = std::hash<std::thread::id>()(std::this_thread::get_id());
    return static_cast<uint64_t>(now) ^ (static_cast<uint64_t>(tid) << 1);
}

//------------------------------------------------------------------------------
// Constructor: initialize PCG state + one-time ziggurat tables
//------------------------------------------------------------------------------
RngEngine::RngEngine(const uint64_t seed) : state_(0), increment_(seed << 1 | 1) {
    // Advance state at least once
    state_ = seed + increment_;
    state_ = state_ * 6364136223846793005ULL + increment_;
}

//------------------------------------------------------------------------------
// nextUInt32(): PCG-XSH-RR 32-bit generator
//------------------------------------------------------------------------------
uint32_t RngEngine::nextUInt32() {
    const uint64_t old = state_;
    state_ = old * 6364136223846793005ULL + increment_;
    const auto xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
    const auto rot = static_cast<uint32_t>(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}


//------------------------------------------------------------------------------
// sampleBoxMuller(): one N(0,1) variate via Ziggurat
//------------------------------------------------------------------------------
float RngEngine::sampleBoxMuller() {
    if (haveSpare_) {
        haveSpare_ = false;
        return static_cast<float>(spare_);
    }
    const double u1 = uniform();
    const double u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 2.0 * M_PI * u2;
    spare_ = r * std::sin(theta);
    haveSpare_ = true;
    return static_cast<float>(r * std::cos(theta));
}

//------------------------------------------------------------------------------
// nextUniform(): convert top-32 bits of nextUInt32() into [0,1)
//------------------------------------------------------------------------------
double RngEngine::uniform() {
    return nextUInt32() * (1.0 / 4294967296.0);
}

//------------------------------------------------------------------------------
// nextNormal(mean,stddev): scale a standard Ziggurat sample
//------------------------------------------------------------------------------
double RngEngine::normal(const double mean, const double stddev) {
    assert(stddev >= 0.0 && "Normal stddev must be non-negative");
    return mean + stddev * static_cast<double>(sampleBoxMuller());
}

//------------------------------------------------------------------------------
// sampleGammaShape1(shape): Marsaglia–Tsang for Gamma(shape,1)
//------------------------------------------------------------------------------
double RngEngine::sampleGammaShape1(const double shape) {
    if (shape < 1.0) {
        // Use a boostrap trick for shape<1
        const double u = uniform();
        return sampleGammaShape1(shape + 1.0) * std::pow(u, 1.0 / shape);
    }
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    while (true) {
        const double x = sampleBoxMuller();
        double v = 1.0 + c * x;
        if (v <= 0) continue;
        v = v * v * v;
        const double u = uniform();
        if (u < 1.0 - 0.0331 * x * x * x * x) {
            return d * v;
        }
        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) {
            return d * v;
        }
    }
}

//------------------------------------------------------------------------------
// nextGamma(shape,scale): scale the unit Gamma by 'scale'
//------------------------------------------------------------------------------
double RngEngine::gamma(const double shape, const double scale) {
    assert(shape > 0.0 && "Gamma shape must be positive");
    assert(scale > 0.0 && "Gamma scale must be positive");
    return sampleGammaShape1(shape) * scale;
}


//------------------------------------------------------------------------------
// nextNegBinomial(r,p): Gamma(r,(1-p)/p) → Poisson(λ) mix
//------------------------------------------------------------------------------

inline int poissonKnuth(RngEngine& rng, const double lambda) {
    const double L = std::exp(-lambda);
    int k = 0;
    double t = 1.0;
    do {
        ++k;
        t *= rng.uniform();
    }
    while (t > L);
    return k - 1;
}


// Atkinson’s rejection for λ ≥ 30
inline int poissonAtkinson(RngEngine& rng, const double lambda) {
    const double c = 0.767 - 3.36 / lambda;
    const double beta = M_PI / std::sqrt(3.0 * lambda);
    const double alpha = beta * lambda;
    const double k = std::log(c) - lambda - std::log(beta);

    while (true) {
        const double u = rng.uniform();
        const double x = (alpha - std::log((1.0 - u) / u)) / beta;
        const int n = static_cast<int>(std::floor(x + 0.5));
        if (n < 0) continue;
        const double v = rng.uniform();
        const double y = alpha - beta * x;
        const double lhs = y + std::log(v / (1.0 + std::exp(y)) * (1.0 + std::exp(y)));
        const double rhs = k + n * std::log(lambda) - std::lgamma(n + 1.0);
        if (lhs <= rhs) return n;
    }
}

int RngEngine::negBinomial(const double r, const double mu) {
    assert(r > 0.0 && "Negative-binomial r must be positive");
    assert(mu >= 0.0 && "Negative-binomial mu must be non-negative");

    const double scale = mu / r;
    const double lambda = sampleGammaShape1(r) * scale;


    return poissonKnuth(*this, lambda);
    //
    // if (lambda < 30.0)
    //     return poissonKnuth(*this, lambda);
    // return poissonAtkinson(*this, lambda);
}
