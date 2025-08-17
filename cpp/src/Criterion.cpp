#include "Criterion.h"
#include <cassert>

using namespace eventide;

// OffspringCriterion
OffspringCriterion::OffspringCriterion(const int minOff, const int maxOff): minOff_(minOff), maxOff_(maxOff) {
    assert(minOff >= 0 && maxOff >= minOff);
}

bool OffspringCriterion::checkRoot(const int nRoot) noexcept {
    ok_ = nRoot >= minOff_ && nRoot <= maxOff_;
    return ok_;
}

bool OffspringCriterion::finalPassed() const noexcept {
    return ok_;
}

std::unique_ptr<Criterion> OffspringCriterion::clone() const noexcept {
    return std::make_unique<OffspringCriterion>(*this);
}

// IntervalCriterion
IntervalCriterion::IntervalCriterion(const double tMin, const double tMax, const int minAllowed, const int maxAllowed)
    : tMin_(tMin), tMax_(tMax), minAllowed_(minAllowed), maxAllowed_(maxAllowed) {
    assert(tMin <= tMax && minAllowed >= 0 && maxAllowed >= minAllowed);
}

void IntervalCriterion::registerTime(const double t) noexcept {
    if (t >= tMin_ && t <= tMax_) count_++;
}

bool IntervalCriterion::earlyReject() const noexcept {
    return count_ > maxAllowed_;
}

bool IntervalCriterion::finalPassed() const noexcept {
    return count_ >= minAllowed_ && count_ <= maxAllowed_;
}

std::unique_ptr<Criterion> IntervalCriterion::clone() const noexcept {
    return std::make_unique<IntervalCriterion>(*this);
}

// CriterionGroup
CriterionGroup::CriterionGroup(const std::vector<std::unique_ptr<Criterion>>& criteria) {
    criteria_.reserve(criteria.size());
    for (auto& c : criteria)
        criteria_.push_back(std::move(c->clone()));
}

CriterionGroup::CriterionGroup(const std::vector<std::shared_ptr<Criterion>>& criteria) {
    criteria_.reserve(criteria.size());
    for (auto& c : criteria)
        criteria_.push_back(std::move(c->clone()));
}

CriterionGroup::CriterionGroup(const CriterionGroup& other) {
    for (auto& c : other.criteria_)
        criteria_.push_back(std::move(c->clone()));
}

bool CriterionGroup::checkRoot(const int nRoot) noexcept {
    bool passed = true;
    for (const auto& c : criteria_)
        passed = passed && c->checkRoot(nRoot);
    return passed;
}

void CriterionGroup::registerTime(const double t) noexcept {
    for (const auto& c : criteria_)
        c->registerTime(t);
}

bool CriterionGroup::earlyReject() const noexcept {
    for (const auto& c : criteria_)
        if (c->earlyReject()) return true;
    return false;
}

bool CriterionGroup::finalPassed() const noexcept {
    for (const auto& c : criteria_)
        if (!c->finalPassed())
            return false;
    return true;
}

std::unique_ptr<Criterion> CriterionGroup::clone() const noexcept {
    return std::make_unique<CriterionGroup>(*this);
}

void CriterionGroup::reset() noexcept {
    for (const auto& c : criteria_)
        c->reset();
}


