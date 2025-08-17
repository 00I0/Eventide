#pragma once
/**
 * @file Criterion.h
 * @brief Acceptance/rejection criteria interfaces and composites.
 */
#include <memory>
#include <vector>

namespace eventide {
    /**
     * @brief Base interface for acceptance criteria.
     */
    class Criterion {
    public:
        virtual ~Criterion() = default;

        /** @brief initial check on the root infection; return false → reject */
        virtual bool checkRoot(int nRoot) noexcept = 0;

        /** @brief record a new infection at time t */
        virtual void registerTime(double t) noexcept = 0;

        /** @brief early‑stop if true → reject immediately */
        virtual bool earlyReject() const noexcept = 0;

        /** @brief final acceptance check at the end of the simulation */
        virtual bool finalPassed() const noexcept = 0;

        /** @brief reset internal state before reusing */
        virtual void reset() noexcept = 0;

        /** @brief clone for per‑workercontext isolation */
        virtual std::unique_ptr<Criterion> clone() const noexcept = 0;
    };

    /**
     * @brief Require the index case to have [min, max] children.
     */
    class OffspringCriterion final : public Criterion {
    public:
        OffspringCriterion(int minOff, int maxOff);

        bool checkRoot(int nRoot) noexcept override;
        void registerTime(double) noexcept override {}
        bool earlyReject() const noexcept override { return false; }
        bool finalPassed() const noexcept override;
        void reset() noexcept override { ok_ = false; }

        std::unique_ptr<Criterion> clone() const noexcept override;

    private:
        int minOff_, maxOff_;
        bool ok_ = false;
    };

    /**
     * @brief Count infections in [tMin, tMax]; require minAllowed ≤ count ≤ maxAllowed.
     */
    class IntervalCriterion final : public Criterion {
    public:
        IntervalCriterion(double tMin, double tMax, int minAllowed, int maxAllowed);

        bool checkRoot(int) noexcept override { return true; }
        void registerTime(double t) noexcept override;
        bool earlyReject() const noexcept override;
        bool finalPassed() const noexcept override;
        void reset() noexcept override { count_ = 0; }

        std::unique_ptr<Criterion> clone() const noexcept override;

    private:
        double tMin_, tMax_;
        int minAllowed_, maxAllowed_, count_ = 0;
    };

    /**
     * @brief Composite of multiple Criterion, applied in sequence.
     */
    class CriterionGroup final : public Criterion {
    public:
        explicit CriterionGroup(const std::vector<std::unique_ptr<Criterion>>& criteria);
        explicit CriterionGroup(const std::vector<std::shared_ptr<Criterion>>& criteria);
        CriterionGroup(const CriterionGroup& other);

        bool checkRoot(int nRoot) noexcept override;
        void registerTime(double t) noexcept override;
        bool earlyReject() const noexcept override;
        bool finalPassed() const noexcept override;
        void reset() noexcept override;

        std::unique_ptr<Criterion> clone() const noexcept override;

    private:
        std::vector<std::unique_ptr<Criterion>> criteria_;
    };
}

