#include "Collector.h"
#include <cassert>
#include <algorithm>

using namespace eventide;

// TimeMatrixCollector
TimeMatrixCollector::TimeMatrixCollector(const int T, const int cutoffDay):
    T_(T), cutoffDay_(cutoffDay), mat_(T + 2, std::vector<long>(T + 2, 0)), maxTime_(0), firstAfter_(cutoffDay + 1) {
    assert(cutoffDay <= T);
}

TimeMatrixCollector::TimeMatrixCollector(const TimeMatrixCollector& o):
    T_(o.T_), cutoffDay_(o.cutoffDay_), mat_(o.mat_), maxTime_(0), firstAfter_(cutoffDay_ + 1) {}

void TimeMatrixCollector::registerTime(const double t) {
    const int day = std::clamp(static_cast<int>(std::floor(t)), 0, T_ + 1);
    if (day > maxTime_) maxTime_ = day;
    if (day > cutoffDay_ && day < firstAfter_) firstAfter_ = day;
}

void TimeMatrixCollector::merge(const DataCollector& other) {
    auto const& o = dynamic_cast<TimeMatrixCollector const&>(other);
    for (int i = 0; i <= T_; i++)
        for (int j = 0; j <= T_; j++)
            mat_[i][j] += o.mat_[i][j];
}

void TimeMatrixCollector::reset() {
    maxTime_ = 0;
    firstAfter_ = cutoffDay_ + 1;
}

void TimeMatrixCollector::save(const TrajectoryResult trajectoryResult) {
    if (trajectoryResult == TrajectoryResult::CAPPED_AT_T_RUN || trajectoryResult == TrajectoryResult::ACCEPTED)
        mat_[maxTime_][firstAfter_] += 1;

    reset();
}


// DrawHistogramCollector
DrawHistogramCollector::DrawHistogramCollector(const std::vector<Parameter>& params, const int nbins):
    params_(params),
    nbins_(nbins),
    hist_(5, std::vector<long>(nbins, 0)),
    R0Bin_(0), rBin_(0), kBin_(0), alphaBin_(0), thetaBin_(0) {
    assert(nbins > 0);
    assert(params.size() == 5);
}

DrawHistogramCollector::DrawHistogramCollector(const DrawHistogramCollector& o):
    params_(o.params_),
    nbins_(o.nbins_),
    hist_(o.hist_),
    R0Bin_(0), rBin_(0), kBin_(0), alphaBin_(0), thetaBin_(0) {}

void DrawHistogramCollector::recordDraw(const Draw& draw) {
    double lo = params_[static_cast<int>(DrawID::R0)].min, hi = params_[static_cast<int>(DrawID::R0)].max;
    R0Bin_ = hi == lo ? 0 : std::min(static_cast<int>((draw.R0 - lo) / (hi - lo) * nbins_), nbins_ - 1);

    lo = params_[static_cast<int>(DrawID::k)].min, hi = params_[static_cast<int>(DrawID::k)].max;
    kBin_ = hi == lo ? 0 : std::min(static_cast<int>((draw.k - lo) / (hi - lo) * nbins_), nbins_ - 1);

    lo = params_[static_cast<int>(DrawID::r)].min, hi = params_[static_cast<int>(DrawID::r)].max;
    rBin_ = hi == lo ? 0 : std::min(static_cast<int>((draw.r - lo) / (hi - lo) * nbins_), nbins_ - 1);

    lo = params_[static_cast<int>(DrawID::alpha)].min, hi = params_[static_cast<int>(DrawID::alpha)].max;
    alphaBin_ = hi == lo ? 0 : std::min(static_cast<int>((draw.alpha - lo) / (hi - lo) * nbins_), nbins_ - 1);

    lo = params_[static_cast<int>(DrawID::theta)].min, hi = params_[static_cast<int>(DrawID::theta)].max;
    thetaBin_ = hi == lo ? 0 : std::min(static_cast<int>((draw.theta - lo) / (hi - lo) * nbins_), nbins_ - 1);
}

void DrawHistogramCollector::merge(const DataCollector& other) {
    auto const& o = dynamic_cast<const DrawHistogramCollector&>(other);

    for (size_t i = 0; i < hist_.size(); ++i) {
        assert(o.hist_[i].size() == hist_[i].size());
        for (size_t b = 0; b < hist_[i].size(); ++b)
            hist_[i][b] += o.hist_[i][b];
    }
}


void DrawHistogramCollector::reset() {
    R0Bin_ = 0;
    rBin_ = 0;
    kBin_ = 0;
    alphaBin_ = 0;
    thetaBin_ = 0;
}

void DrawHistogramCollector::save(const TrajectoryResult trajectoryResult) {
    hist_[static_cast<int>(DrawID::R0)][R0Bin_] += 1;
    hist_[static_cast<int>(DrawID::k)][kBin_] += 1;
    hist_[static_cast<int>(DrawID::r)][rBin_] += 1;
    hist_[static_cast<int>(DrawID::alpha)][alphaBin_] += 1;
    hist_[static_cast<int>(DrawID::theta)][thetaBin_] += 1;

    reset();
}


// JointHeatmapCollector
JointHeatmapCollector::JointHeatmapCollector(const double R0min, const double R0max, const double rmin,
                                             const double rmax, const int bins):
    R0min_(R0min), R0max_(R0max), rmin_(rmin), rmax_(rmax), bins_(bins), heat_(bins, std::vector<long>(bins, 0)),
    binI_(0), binJ_(0) {
    assert(bins > 0);
}


JointHeatmapCollector::JointHeatmapCollector(const JointHeatmapCollector& o):
    R0min_(o.R0min_), R0max_(o.R0max_), rmin_(o.rmin_), rmax_(o.rmax_), bins_(o.bins_), heat_(o.heat_), binI_(0),
    binJ_(0) {}


void JointHeatmapCollector::recordDraw(const Draw& draw) {
    const double R0 = draw.R0, r = draw.r;
    if (R0 < R0min_ || R0 > R0max_ || r < rmin_ || r > rmax_) return;

    binI_ = std::min(static_cast<int>((R0 - R0min_) / (R0max_ - R0min_) * bins_), bins_ - 1);
    binJ_ = std::min(static_cast<int>((r - rmin_) / (rmax_ - rmin_) * bins_), bins_ - 1);
}

void JointHeatmapCollector::merge(const DataCollector& other) {
    const auto& o = dynamic_cast<const JointHeatmapCollector&>(other);
    assert(o.heat_.size() == heat_.size());
    for (int i = 0; i < bins_; i++)
        for (int j = 0; j < bins_; j++)
            heat_[i][j] += o.heat_[i][j];
}

void JointHeatmapCollector::reset() {
    binI_ = 0;
    binJ_ = 0;
}

void JointHeatmapCollector::save(const TrajectoryResult trajectoryResult) {
    heat_[binI_][binJ_] += 1;
    reset();
}


// DerivedMarginalCollector
DerivedMarginalCollector::DerivedMarginalCollector(const Product prod, const double lo, const double hi,
                                                   const int bins):
    prod_(prod), lo_(lo), hi_(hi), bins_(bins), hist_(bins, 0), val_(0.0) {
    assert(bins > 0 && lo < hi);
}

DerivedMarginalCollector::DerivedMarginalCollector(const DerivedMarginalCollector& o):
    prod_(o.prod_), lo_(o.lo_), hi_(o.hi_), bins_(o.bins_), hist_(o.hist_), val_(0.0) {}


void DerivedMarginalCollector::recordDraw(const Draw& draw) {
    if (prod_ == Product::R0_r) val_ = draw.R0 * draw.r;
    else if (prod_ == Product::AlphaTheta) val_ = draw.alpha * draw.theta;
}


void DerivedMarginalCollector::merge(const DataCollector& other) {
    const auto& o = dynamic_cast<const DerivedMarginalCollector&>(other);
    assert(o.hist_.size() == hist_.size());
    for (size_t i = 0; i < hist_.size(); ++i)
        hist_[i] += o.hist_[i];
}

void DerivedMarginalCollector::reset() {
    val_ = 0.0;
}

void DerivedMarginalCollector::save(const TrajectoryResult trajectoryResult) {
    if (val_ < lo_ || val_ > hi_) return;
    const int bin = std::min(static_cast<int>((val_ - lo_) / (hi_ - lo_) * bins_), bins_ - 1);
    hist_[bin] += 1;

    reset();
}


//DataCollectorGroup

DataCollectorGroup::DataCollectorGroup(std::vector<std::unique_ptr<DataCollector>> collectors):
    collectors_(std::move(collectors)) {}

DataCollectorGroup::DataCollectorGroup(const DataCollectorGroup& other) {
    collectors_.reserve(other.collectors_.size());
    for (const auto& collector : other.collectors_)
        collectors_.emplace_back(collector->clone());
}


void DataCollectorGroup::merge(const DataCollector& other) {
    const auto& o = dynamic_cast<const DataCollectorGroup&>(other);
    assert(o.collectors_.size() == collectors_.size());
    for (size_t i = 0; i < collectors_.size(); ++i)
        collectors_[i]->merge(*o.collectors_[i]);
}






