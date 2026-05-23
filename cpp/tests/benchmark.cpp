#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
// Include your Eventide headers:
#include "Parameter.h"
#include "Sampler.h"
#include "Scenario.h"
#include "Criterion.h"
#include "Collector.h"
#include "Simulator.h"

using namespace eventide;

template <typename T>
static double quantile(std::vector<T> v, double q) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const double pos = q * (v.size() - 1);
    const size_t i = static_cast<size_t>(std::floor(pos));
    const double frac = pos - i;
    if (i + 1 < v.size()) return v[i] * (1.0 - frac) + v[i + 1] * frac;
    return static_cast<double>(v[i]);
}

double run_simulator() {
    // --- 1) Define parameters ---
    std::vector params = {
        Parameter("R0", 0.25, 15),
        Parameter("r", 0.01, 0.99),
        Parameter("k", 0.2, 10),
        Parameter("alpha", 0.01, 20),
        Parameter("theta", 0.01, 20)
    };


    CompiledExpression validator("(R0 * r < 3) "
        "and (1 < alpha * theta) and (alpha * theta < 28) "
        "and (R0 / k < 1.2) and (R0 * r / k < 0.4)"
        "and ((k / (k + R0 * r)) ^ k > 0.05) and ((k / (k + R0 * r)) ^ k < 0.95)"
        "and ((R0 * r) ^ (1 / alpha) - 1) / theta < 0.1000001"
        "and (sqrt(alpha) * theta <= 21)"
    );

    RngEngine rng;
    LatinHypercubeSampler sampler(params, rng, true);

    // --- 2) Scenario: empty ---
    Scenario scenario(std::vector<ParameterChangePoint>{});

    // --- 3) Acceptance criteria ---
    std::vector<std::unique_ptr<Criterion>> criteria;
    criteria.emplace_back(std::make_unique<OffspringCriterion>(2, 5));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(IntervalCriterion(0.0, 7.7839, 0, 2)));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(IntervalCriterion(0.0, 45.0, 8, 12)));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(IntervalCriterion(7.7839, 19.7480, 2, 4)));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(IntervalCriterion(19.7480, 22.9193, 1, 3)));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(IntervalCriterion(22.9193, 26.5229, 0, 1)));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(IntervalCriterion(26.5229, 29.6942, 2, 3)));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(IntervalCriterion(29.6942, 32.8654, 1, 2)));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(IntervalCriterion(32.8654, 39.2079, 0, 1)));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(IntervalCriterion(39.2079, 44.9737, 0, 0)));


    CriterionGroup critGroup(std::move(criteria));

    // --- 4) Data collectors ---
    int T_run = 55;
    int N_TRAJ = 1'000'000'000;
    int minReq = 2000;

    auto tm_col = std::make_shared<InfectionTimeCollector>();

    std::vector<std::shared_ptr<DataCollector>> collectors;
    collectors.push_back(tm_col);
    DataCollectorGroup collGroup(std::move(collectors));

    // --- 5) Run simulation ---
    int chunk_size = 100'000;
    int max_cases = 1000;
    int max_workers = 13;
    auto start = std::chrono::high_resolution_clock::now();

    Simulator sim(sampler, scenario, critGroup, collGroup, N_TRAJ, minReq, chunk_size,
                  T_run, max_cases, max_workers, validator);
    sim.run();

    auto stop = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration<double>(stop - start).count();

    std::cout << "  |  Runtime: " << std::fixed << std::setprecision(4) << runtime << " seconds ";
    int accepted = static_cast<InfectionTimeCollector*>(sim.collectors().at(0).get())->infectionTimes().size();
    collGroup.at(0)->recordDraw({0, 0, 0, 0, 0});
    std::cout << " |  Trajectories accepted: " << accepted << " / " << N_TRAJ << std::endl;

    return runtime;
}

int main() {
    constexpr int REPEATS = 10;
    std::vector<double> runtimes;
    for (int i = 0; i < REPEATS; ++i) {
        std::cout << "\t" << i << "\t";
        runtimes.push_back(run_simulator());
    }


    const double mean = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();

    double var = 0.0;
    if (runtimes.size() > 1) {
        for (const double x : runtimes) var += (x - mean) * (x - mean);
        var /= runtimes.size() - 1;
    }
    const double sd = std::sqrt(var);

    std::vector<double> copy_rt = runtimes;
    const double p50 = quantile(copy_rt, 0.50);
    const double p95 = quantile(copy_rt, 0.95);
    const double minv = *std::min_element(copy_rt.begin(), copy_rt.end());
    const double maxv = *std::max_element(copy_rt.begin(), copy_rt.end());


    std::cout << "\n=== Runtime stats over " << REPEATS << " runs ===\n"
        << "min   : " << std::fixed << std::setprecision(4) << minv << " s\n"
        << "median: " << p50 << " s\n"
        << "p95   : " << p95 << " s\n"
        << "max   : " << maxv << " s\n"
        << "mean  : " << mean << " s  (sd = " << sd << " s)\n";


    return 0;
}
