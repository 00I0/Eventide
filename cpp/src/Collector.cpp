#include "Collector.h"
#include <cassert>
#include <algorithm>

using namespace eventide;

// TimeMatrixCollector
TimeMatrixCollector::TimeMatrixCollector(const int T, const int cutoffDay):
    T_(T), cutoffDay_(cutoffDay), mat_(T + 2, std::vector<long>(T + 2, 0)), maxTime_(0), firstAfter_(T + 1) {
    assert(cutoffDay <= T);
}

TimeMatrixCollector::TimeMatrixCollector(const TimeMatrixCollector& other):
    T_(other.T_), cutoffDay_(other.cutoffDay_), mat_(other.mat_), maxTime_(0), firstAfter_(other.T_ + 1) {}

void TimeMatrixCollector::registerTime(const double parentInfectionTime, const double newInfectionTime) {
    const int day = std::clamp(static_cast<int>(std::floor(newInfectionTime)), 0, T_ + 1);
    if (day > maxTime_) maxTime_ = day;
    if (day > cutoffDay_ && day < firstAfter_) firstAfter_ = day;
}

void TimeMatrixCollector::merge(const DataCollector& other) {
    auto const& o = dynamic_cast<TimeMatrixCollector const&>(other);
    for (int i = 0; i <= T_ + 1; i++)
        for (int j = 0; j <= T_ + 1; j++)
            mat_[i][j] += o.mat_[i][j];
}

void TimeMatrixCollector::reset() {
    maxTime_ = 0;
    firstAfter_ = T_ + 1;
}

void TimeMatrixCollector::save(const TrajectoryResult trajectoryResult) {
    if (trajectoryResult == TrajectoryResult::CAPPED_AT_T_RUN || trajectoryResult == TrajectoryResult::ACCEPTED)
        mat_[maxTime_][firstAfter_] += 1;

    reset();
}

//Hist1D
Hist1D::Hist1D(const CompiledExpression& expression, const int bins, const double lo, const double hi):
    expression_(expression), bins_(bins), lo_(lo), hi_(hi), hist_(bins, 0), val_(0.0) {}


Hist1D::Hist1D(const Hist1D& other): expression_(other.expression_), bins_(other.bins_), lo_(other.lo_),
                                     hi_(other.hi_), hist_(other.hist_), val_(0.0) {}

void Hist1D::recordDraw(const Draw& draw) {
    val_ = expression_.eval(draw);
}

void Hist1D::save(TrajectoryResult trajectoryResult) {
    if (val_ < lo_ || val_ > hi_) return;
    const int bin = std::min(static_cast<int>((val_ - lo_) / (hi_ - lo_) * bins_), bins_ - 1);
    hist_[bin] += 1;

    reset();
}

void Hist1D::merge(const DataCollector& other) {
    const auto& o = dynamic_cast<const Hist1D&>(other);
    assert(o.hist_.size() == hist_.size());
    for (size_t i = 0; i < hist_.size(); ++i)
        hist_[i] += o.hist_[i];
}


//Hist2D
Hist2D::Hist2D(const CompiledExpression& expressionX, const CompiledExpression& expressionY,
               const int bins, const double loX, const double hiX, const double loY, const double hiY):
    expressionX_(expressionX), expressionY_(expressionY), bins_(bins), loX_(loX), hiX_(hiX), loY_(loY), hiY_(hiY),
    hist_(bins, std::vector<long>(bins, 0)), valX_(0.0), valY_(0.0) {}

Hist2D::Hist2D(const Hist2D& other): expressionX_(other.expressionX_), expressionY_(other.expressionY_),
                                     bins_(other.bins_), loX_(other.loX_), hiX_(other.hiX_), loY_(other.loY_),
                                     hiY_(other.hiY_), hist_(other.hist_), valX_(0.0), valY_(0.0) {}

void Hist2D::recordDraw(const Draw& draw) {
    valX_ = expressionX_.eval(draw);
    valY_ = expressionY_.eval(draw);
}

void Hist2D::save(const TrajectoryResult trajectoryResult) {
    if (valX_ < loX_ || valX_ > hiX_ || valY_ < loY_ || valY_ > hiY_) return;

    const int binX = std::min(static_cast<int>((valX_ - loX_) / (hiX_ - loX_) * bins_), bins_ - 1);
    const int binY = std::min(static_cast<int>((valY_ - loY_) / (hiY_ - loY_) * bins_), bins_ - 1);
    hist_[binX][binY] += 1;
}

void Hist2D::merge(const DataCollector& other) {
    const auto& o = dynamic_cast<const Hist2D&>(other);
    assert(o.hist_.size() == hist_.size());
    for (int i = 0; i < bins_; i++)
        for (int j = 0; j < bins_; j++)
            hist_[i][j] += o.hist_[i][j];
}


//DataCollectorGroup
DataCollectorGroup::DataCollectorGroup(const std::vector<std::unique_ptr<DataCollector>>& collectors) {
    collectors_.reserve(collectors.size());
    for (const auto& collector : collectors)
        collectors_.emplace_back(collector->clone());
}

DataCollectorGroup::DataCollectorGroup(const std::vector<std::shared_ptr<DataCollector>>& collectors) {
    collectors_.reserve(collectors.size());
    for (const auto& collector : collectors)
        collectors_.emplace_back(collector->clone());
}

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


// DrawCollector
void DrawCollector::recordDraw(const Draw& draw) {
    currentDraw_[static_cast<size_t>(DrawID::R0)] = draw.R0;
    currentDraw_[static_cast<size_t>(DrawID::k)] = draw.k;
    currentDraw_[static_cast<size_t>(DrawID::r)] = draw.r;
    currentDraw_[static_cast<size_t>(DrawID::alpha)] = draw.alpha;
    currentDraw_[static_cast<size_t>(DrawID::theta)] = draw.theta;
}

void DrawCollector::save(const TrajectoryResult trajectoryResult) {
    if (trajectoryResult == TrajectoryResult::CAPPED_AT_T_RUN || trajectoryResult == TrajectoryResult::ACCEPTED)
        draws_.push_back(currentDraw_);
}

void DrawCollector::merge(const DataCollector& other) {
    const auto& o = dynamic_cast<const DrawCollector&>(other);

    draws_.reserve(draws_.size() + o.draws_.size());
    draws_.insert(draws_.end(), o.draws_.begin(), o.draws_.end());
}


//ActiveSetSizeCollector
ActiveSetSizeCollector::ActiveSetSizeCollector(const double collectionTime):
    collectionTime_(collectionTime) {}

void ActiveSetSizeCollector::merge(const DataCollector& other) {
    const auto& o = dynamic_cast<const ActiveSetSizeCollector&>(other);

    activeSetSizes_.reserve(activeSetSizes_.size() + o.activeSetSizes_.size());
    activeSetSizes_.insert(activeSetSizes_.end(), o.activeSetSizes_.begin(), o.activeSetSizes_.end());
}

void ActiveSetSizeCollector::reset() {
    currentActiveSetSize = 0;
}

void ActiveSetSizeCollector::registerTime(const double parentInfectionTime, const double newInfectionTime) {
    if (parentInfectionTime <= collectionTime_ && collectionTime_ < newInfectionTime)
        currentActiveSetSize += 1;
}

void ActiveSetSizeCollector::save(const TrajectoryResult trajectoryResult) {
    if (trajectoryResult == TrajectoryResult::CAPPED_AT_T_RUN || trajectoryResult == TrajectoryResult::ACCEPTED)
        activeSetSizes_.push_back(currentActiveSetSize);

    reset();
}


//InfectionTimeCollector
void InfectionTimeCollector::merge(const DataCollector& other) {
    const auto& o = dynamic_cast<const InfectionTimeCollector&>(other);

    allInfectionTimes_.reserve(allInfectionTimes_.size() + o.allInfectionTimes_.size());
    allInfectionTimes_.insert(allInfectionTimes_.end(), o.allInfectionTimes_.begin(), o.allInfectionTimes_.end());
}


void InfectionTimeCollector::registerTime(double parentInfectionTime, const double newInfectionTime) {
    currentInfectionTimes_.push_back(newInfectionTime);
}

void InfectionTimeCollector::reset() {
    currentInfectionTimes_.clear();
}

void InfectionTimeCollector::save(TrajectoryResult trajectoryResult) {
    allInfectionTimes_.push_back(currentInfectionTimes_);

    reset();
}
