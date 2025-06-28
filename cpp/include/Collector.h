#pragma once
#include <typeindex>
#include <vector>

#include "CompiledExpression.h"
#include "Parameter.h"
#include "Sampler.h"
#include "TrajectoryResult.h"

namespace eventide {
    /**
     * @brief Base interface for streaming & final data collection.
     */
    class DataCollector {
    public:
        virtual ~DataCollector() = default;

        /// Reset per‐trajectory temporary state.
        virtual void reset() = 0;

        /// Called on each new infection time (optional).
        virtual void registerTime(double t) = 0;

        /// Or: record parameters of the accepted trajectory.
        virtual void recordDraw(const Draw& params) = 0;

        /// Commit per‐trajectory state into the long‐term storage.
        virtual void save(TrajectoryResult trajectoryResult) = 0;

        /// Merge another collector of the same concrete type into this one.
        /// (merges only the long-term storage).
        virtual void merge(const DataCollector& other) = 0;

        /// “Virtual copy constructor” for making a fresh, empty instance.
        virtual std::unique_ptr<DataCollector> clone() const = 0;
    };

    /**
     * @brief (T+2)×(T+2) matrix of [floor(finalTime)][floor(firstAfterCutoff)] counts.
     */
    class TimeMatrixCollector final : public DataCollector {
    public:
        /// @param T          max day to tabulate
        /// @param cutoffDay  day X for "first after" detection
        explicit TimeMatrixCollector(int T, int cutoffDay);
        TimeMatrixCollector(const TimeMatrixCollector& o);

        std::unique_ptr<DataCollector> clone() const override {
            return std::make_unique<TimeMatrixCollector>(*this);
        }

        void reset() override;
        void registerTime(double t) override;
        void recordDraw(const Draw& draw) override {}
        void merge(const DataCollector& other) override;
        void save(TrajectoryResult trajectoryResult) override;

        const std::vector<std::vector<long>>& matrix() const { return mat_; }

    private:
        int T_;
        int cutoffDay_;
        std::vector<std::vector<long>> mat_;

        // per-trajectory state
        int maxTime_;
        int firstAfter_;
    };


    class Hist1D final : public DataCollector {
    public:
        explicit Hist1D(const CompiledExpression& expression, int bins, double lo, double hi);
        Hist1D(const Hist1D& other);

        std::unique_ptr<DataCollector> clone() const override { return std::make_unique<Hist1D>(*this); }

        void reset() override {}
        void registerTime(double t) override {}
        void recordDraw(const Draw& draw) override;
        void merge(const DataCollector& other) override;
        void save(TrajectoryResult trajectoryResult) override;

        std::vector<long> histogram() const { return hist_; }

    private:
        CompiledExpression expression_;
        int bins_;
        double lo_, hi_;
        std::vector<long> hist_;
        double val_; // per trajectory
    };

    class Hist2D final : public DataCollector {
    public:
        explicit Hist2D(const CompiledExpression& expressionX, const CompiledExpression& expressionY,
                        int bins, double loX, double hiX, double loY, double hiY);
        Hist2D(const Hist2D& other);

        std::unique_ptr<DataCollector> clone() const override { return std::make_unique<Hist2D>(*this); }

        void reset() override {}
        void registerTime(double t) override {}
        void recordDraw(const Draw& draw) override;
        void merge(const DataCollector& other) override;
        void save(TrajectoryResult trajectoryResult) override;

        std::vector<std::vector<long>> histogram() const { return hist_; }

    private:
        CompiledExpression expressionX_, expressionY_;
        int bins_;
        double loX_, hiX_, loY_, hiY_;
        std::vector<std::vector<long>> hist_;
        double valX_, valY_; // per trajectory
    };

    class DataCollectorGroup final : public DataCollector {
    public:
        /// Takes ownership of each collector
        explicit DataCollectorGroup(std::vector<std::unique_ptr<DataCollector>> collectors);
        DataCollectorGroup(const DataCollectorGroup& other);

        std::unique_ptr<DataCollector> clone() const override {
            return std::make_unique<DataCollectorGroup>(*this);
        }

        void reset() override {
            for (const auto& c : collectors_) c->reset();
        }

        void registerTime(const double t) override {
            for (const auto& c : collectors_) c->registerTime(t);
        }

        void recordDraw(const Draw& params) override {
            for (const auto& c : collectors_) c->recordDraw(params);
        }

        void merge(const DataCollector& other) override;

        void save(const TrajectoryResult trajectoryResult) override {
            for (const auto& c : collectors_) c->save(trajectoryResult);
        }

        template <typename T>
        T* get() const {
            for (auto& c : collectors_) if (auto p = dynamic_cast<T*>(c.get())) return p;
            return nullptr;
        }

        DataCollector* findByType(const std::type_index id) const {
            for (auto& up : collectors_)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpotentially-evaluated-expression"

                if (std::type_index(typeid(*up)) == id) return up.get();

#pragma clang diagnostic pop
            return nullptr;
        }

    private:
        std::vector<std::unique_ptr<DataCollector>> collectors_;
    };
}
