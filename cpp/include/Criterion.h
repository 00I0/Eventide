#pragma once
#include <memory>
#include <vector>

namespace eventide {
    /**
     * @brief Base interface for acceptance criteria.
     */
    class Criterion {
    public:
        virtual ~Criterion() = default;

        /// Called once, immediately after drawing root’s offspring.
        virtual bool checkRoot(int nRoot) = 0;

        /// Called on each new infection time.
        virtual void registerTime(double t) = 0;

        /// Immediately after registerTime, if true → early reject.
        virtual bool earlyReject() const = 0;

        /// After the trajectory ends, final acceptance?
        virtual bool finalPassed() const = 0;

        /// Clone for per-trajectory instantiation.
        virtual std::unique_ptr<Criterion> clone() const = 0;

        /// Resets the counters
        virtual void reset() = 0;
    };

    /**
     * @brief Require the index case to have [min,max] children.
     */
    class OffspringCriterion final : public Criterion {
    public:
        OffspringCriterion(int minOff, int maxOff);

        bool checkRoot(int nRoot) override;
        void registerTime(double) override {}
        bool earlyReject() const override { return false; }
        bool finalPassed() const override;
        void reset() override { ok_ = false; }

        std::unique_ptr<Criterion> clone() const override;

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

        bool checkRoot(int) override { return true; }
        void registerTime(double t) override;
        bool earlyReject() const override;
        bool finalPassed() const override;
        void reset() override { count_ = 0; }

        std::unique_ptr<Criterion> clone() const override;

    private:
        double tMin_, tMax_;
        int minAllowed_, maxAllowed_, count_ = 0;
    };

    class CriterionGroup final : public Criterion {
    public:
        explicit CriterionGroup(const std::vector<std::unique_ptr<Criterion>>& criteria);
        CriterionGroup(const CriterionGroup& other);

        bool checkRoot(int nRoot) override;
        void registerTime(double t) override;
        bool earlyReject() const override;
        bool finalPassed() const override;
        void reset() override;

        std::unique_ptr<Criterion> clone() const override;

    private:
        std::vector<std::unique_ptr<Criterion>> criteria_;
    };
}

