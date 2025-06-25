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

int main() {
    // --- 1) Define parameters ---
    std::vector<Parameter> params = {
        Parameter("R0", 0.25, 15),
        Parameter("r", 0.01, 0.99),
        Parameter("k", 0.2, 10),
        Parameter("alpha", 0.01, 20),
        Parameter("theta", 0.01, 20)
    };

    RngEngine rng;
    LatinHypercubeSampler sampler(params, rng, true);

    // --- 2) Scenario: empty ---
    Scenario scenario(std::vector<ParameterChangePoint>{});

    // --- 3) Acceptance criteria ---
    std::vector<std::unique_ptr<Criterion>> criteria;
    criteria.emplace_back(std::make_unique<OffspringCriterion>(2, 5));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(0.0, 29.0, 9, 11));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(29.0, 32.0, 1, 1));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(32.0, 42.0, 0, 0));
    criteria.emplace_back(std::make_unique<IntervalCriterion>(42.0, 45.0, 1, 1));
    CriterionGroup critGroup(std::move(criteria));

    // --- 4) Data collectors ---
    int T_run = 60;
    int cut_off_day = 45;
    int N_TRAJ = 100'000;

    auto tm_col = std::make_unique<TimeMatrixCollector>(T_run, cut_off_day);
    auto ph_col = std::make_unique<DrawHistogramCollector>(params, 200);
    auto jh_col = std::make_unique<JointHeatmapCollector>(0.25, 15.0, 0.01, 0.99, 50);
    auto dm1_col = std::make_unique<DerivedMarginalCollector>(DerivedMarginalCollector::Product::R0_r, 0.0, 10.0, 200);
    auto dm2_col = std::make_unique<DerivedMarginalCollector>(DerivedMarginalCollector::Product::AlphaTheta, 1.0, 50.0,
                                                              200);

    std::vector<std::unique_ptr<DataCollector>> collectors;
    collectors.push_back(std::move(tm_col));
    collectors.push_back(std::move(ph_col));
    collectors.push_back(std::move(jh_col));
    collectors.push_back(std::move(dm1_col));
    collectors.push_back(std::move(dm2_col));

    DataCollectorGroup collGroup(std::move(collectors));

    // --- 5) Run simulation ---
    int chunk_size = 1'000'000;
    int max_cases = 1000;
    int max_workers = 1; // For profiling, single-thread is best
    auto start = std::chrono::high_resolution_clock::now();

    Simulator sim(sampler, scenario, critGroup, collGroup,
                  N_TRAJ, chunk_size, T_run, max_cases, max_workers, cut_off_day);
    sim.run();

    auto stop = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration<double>(stop - start).count();
    std::cout << "Runtime: " << runtime << " seconds" << std::endl;

    // --- 6) Retrieve results ---
    // Use .get() to access pointers in the DataCollectorGroup (if your API exposes them, or cast from collGroup if needed)
    // Example:
    return 0;
    // const auto
    //     * time_matrix_col = dynamic_cast<TimeMatrixCollector*>(collGroup.collectors()[0].get());
    // const auto* param_hist_col = dynamic_cast<DrawHistogramCollector*>(collGroup.collectors()[1].get());
    // const auto* jh_col_ptr = dynamic_cast<JointHeatmapCollector*>(collGroup.collectors()[2].get());
    // const auto* dm1_col_ptr = dynamic_cast<DerivedMarginalCollector*>(collGroup.collectors()[3].get());
    // const auto* dm2_col_ptr = dynamic_cast<DerivedMarginalCollector*>(collGroup.collectors()[4].get());
    //
    // // Print or dump the main summary:
    // size_t accepted = 0;
    // if (time_matrix_col) {
    //     const auto& mat = time_matrix_col->matrix();
    //     for (const auto& row : mat)
    //         for (int v : row)
    //             accepted += v;
    // }
    // std::cout << "Trajectories accepted: " << accepted << " / " << N_TRAJ << std::endl;
    // if (accepted == 0)
    //     std::cout << "⚠️  All collectors are zero. Try relaxing or removing your acceptance criteria." << std::endl;
    //
    // // You can write out data to file for later inspection (CSV or NPY, if you wish)
    // // Example: write time_matrix to CSV
    // if (time_matrix_col) {
    //     std::ofstream out("time_matrix.csv");
    //     const auto& mat = time_matrix_col->matrix();
    //     for (const auto& row : mat) {
    //         for (size_t i = 0; i < row.size(); ++i) {
    //             out << row[i];
    //             if (i + 1 != row.size()) out << ",";
    //         }
    //         out << "\n";
    //     }
    // }
    //
    // Likewise for param_hist_col, etc.

    return 0;
}
