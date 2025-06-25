// #include "Simulator.h"
// #include "RngEngine.h"
//
// #include <queue>
// #include <algorithm>
// #include <cassert>
// #include <iostream>
// #include <map>
//
// using namespace eventide;
//
//
// namespace {
//     using Params = Draw;
//
//     // pack a draw into Params
//     Params toParams(const std::vector<double>& d) {
//         return Params{d[0], d[1], d[2], d[3], d[4]};
//     }
//
//     // clone one trajectory’s worth of criteria
//     std::vector<std::unique_ptr<Criterion>>
//     cloneCriteria(const std::vector<std::unique_ptr<Criterion>>& proto) {
//         std::vector<std::unique_ptr<Criterion>> v;
//         v.reserve(proto.size());
//         for (auto const& p : proto) v.push_back(p->clone());
//         return v;
//     }
//
//     // apply next change‐point (set or restore) if due
//     inline void applyScenarioIfNeeded(double t,
//                                       Scenario& scenario,
//                                       Params& current,
//                                       const Params& original) {
//         if (!scenario.shouldApply(t)) return;
//         scenario.applyNext(current, original);
//     }
//
//     // the main heap‐driven branching loop
//     inline void runBranching(RngEngine& rng,
//                              Scenario scenario,
//                              std::vector<std::unique_ptr<Criterion>> crits,
//                              const Params& original,
//                              Params& current,
//                              std::priority_queue<double, std::vector<double>, std::greater<>> heap,
//                              double& finalTime,
//                              double& firstAfter,
//                              int T_run,
//                              int maxCases,
//                              int cutoffDay,
//                              const std::vector<std::unique_ptr<DataCollector>>& collectors) {
//         // initialize with root’s children
//         // std::priority_queue<double, std::vector<double>, std::greater<>> heap;
//         // const int nRoot = rng.negBinomial(current.k, current.k / (current.k + current.R0 * current.r));
//         // int cases = nRoot;
//         // for (int i = 0; i < nRoot; ++i)
//         // heap.push(rng.gamma(current.alpha, current.theta));
//
//         int cases = heap.size();
//
//         // branch until done
//         while (!heap.empty() && cases < maxCases) {
//             double t = heap.top();
//             if (t > T_run) break;
//             heap.pop();
//             ++cases;
//
//             finalTime = std::max(finalTime, t);
//             if (t > cutoffDay &&
//                 firstAfter == std::numeric_limits<double>::infinity()) {
//                 firstAfter = t;
//             }
//
//             applyScenarioIfNeeded(t, scenario, current, original);
//
//             // criteria checks
//             bool reject = false;
//             for (auto& c : crits) {
//                 c->registerTime(t);
//                 if (c->earlyReject()) {
//                     reject = true;
//                     break;
//                 }
//             }
//             if (reject) {
//                 while (!heap.empty()) heap.pop();
//                 break;
//             }
//
//             // collectors
//             for (auto& col : collectors) col->registerTime(t);
//
//             // draw next‐gen
//             int nOff = rng.negBinomial(
//                 current.k,
//                 current.k / (current.k + current.R0 * current.r)
//             );
//             for (int i = 0; i < nOff; ++i)
//                 heap.push(t + rng.gamma(current.alpha, current.theta));
//         }
//     }
//
//     // once accepted, dispatch to collectors
//     inline void recordResult(const Params& current,
//                              double finalTime,
//                              double firstAfter,
//                              const std::vector<std::unique_ptr<DataCollector>>& collectors) {
//         std::vector<double> pvec = {
//             current.R0, current.k, current.r,
//             current.alpha, current.theta
//         };
//         for (auto& col : collectors) {
//             if (auto tm = dynamic_cast<TimeMatrixCollector*>(col.get())) {
//                 tm->recordFinal(
//                     int(std::floor(finalTime)),
//                     int(std::floor(firstAfter))
//                 );
//             }
//             else {
//                 col->recordParameters(pvec);
//             }
//         }
//     }
//
//     // top‐level per‐trajectory driver
//     inline void processTrajectory(const std::vector<double>& draw,
//                                   RngEngine& rng,
//                                   const Scenario& scenarioProto,
//                                   const std::vector<std::unique_ptr<Criterion>>& criteriaProto,
//                                   const std::vector<std::unique_ptr<DataCollector>>& collectors,
//                                   int T_run,
//                                   int maxCases,
//                                   int cutoffDay) {
//         auto crits = cloneCriteria(criteriaProto);
//         Scenario scenario = scenarioProto;
//         scenario.reset();
//
//         Params current = toParams(draw);
//         const Params original = current;
//
//         // root‐offspring check
//         int nRoot = rng.negBinomial(
//             current.k,
//             current.k / (current.k + current.R0 * current.r)
//         );
//         if (!crits[0]->checkRoot(nRoot)) return;
//
//         double finalTime = 0.0;
//         double firstAfter = std::numeric_limits<double>::infinity();
//
//         std::priority_queue<double, std::vector<double>, std::greater<>> heap;
//         for (int i = 0; i < nRoot; ++i) {
//             heap.push(rng.gamma(current.alpha, current.theta));
//         }
//
//         runBranching(rng,
//                      std::move(scenario),
//                      std::move(crits),
//                      original,
//                      current,
//                      heap,
//                      finalTime,
//                      firstAfter,
//                      T_run,
//                      maxCases,
//                      cutoffDay,
//                      collectors);
//
//         recordResult(current, finalTime, firstAfter, collectors);
//     }
// }
//
//
// Simulator::Simulator(std::unique_ptr<LatinHypercubeSampler> sampler,
//                      std::unique_ptr<Scenario> scenario,
//                      std::vector<std::unique_ptr<Criterion>> criteria,
//                      std::vector<std::unique_ptr<DataCollector>> collectors,
//                      const int numTrajectories,
//                      const int chunkSize,
//                      const int T_run,
//                      const int maxCases,
//                      const int maxWorkers,
//                      const int cutoffDay)
//     : sampler_(std::move(sampler)),
//       scenarioProto_(std::move(scenario)),
//       criteriaProto_(std::move(criteria)),
//       collectors_(std::move(collectors)),
//       numTrajectories_(numTrajectories),
//       chunkSize_(chunkSize),
//       T_run_(T_run),
//       maxCases_(maxCases),
//       maxWorkers_(maxWorkers),
//       cutoffDay_(cutoffDay) {
//     assert(numTrajectories_ > 0 && chunkSize_ > 0 && T_run_ >= 0);
// }
//
// void Simulator::run() const {
//     const int nChunks = (numTrajectories_ + chunkSize_ - 1) / chunkSize_;
//     std::atomic nextChunk{0};
//
//     std::vector<std::thread> workers;
//     for (int i = 0; i < maxWorkers_; i++) {
//         workers.emplace_back([&]() {
//             // ReSharper disable once CppTooWideScope
//             RngEngine rng;
//             while (true) {
//                 const int idx = nextChunk.fetch_add(1);
//                 if (idx >= nChunks) break;
//                 processChunk(idx, rng);
//             }
//         });
//     }
//
//     for (auto& worker : workers) worker.join();
//     for (const auto& collector : collectors_) collector->finalize();
// }
//
// void Simulator::processChunk(const int chunkIndex, RngEngine& rng) const {
//     const int remain = numTrajectories_ - chunkIndex * chunkSize_;
//     const int blockSz = std::min(chunkSize_, remain);
//     const auto block = sampler_->sampleBlock(blockSz);
//
//     for (auto const& draw : block)
//         processTrajectory(draw, rng, *scenarioProto_, criteriaProto_, collectors_, T_run_, maxCases_, cutoffDay_);
// }
//
//
//
