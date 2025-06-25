#pragma once
#include <typeindex>
#include <vector>

#include "Parameter.h"
#include "Sampler.h"

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
        virtual void save() = 0;

        /// Merge another collector of the same concrete type into this one.
        /// (merges only the long-term storage).
        virtual void merge(const DataCollector& other) = 0;

        /// “Virtual copy constructor” for making a fresh, empty instance.
        virtual std::unique_ptr<DataCollector> clone() const = 0;
    };

    /**
     * @brief (T+1)×(T+1) matrix of [floor(finalTime)][floor(firstAfterCutoff)] counts.
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
        void save() override;

        const std::vector<std::vector<long>>& matrix() const { return mat_; }

    private:
        int T_;
        int cutoffDay_;
        std::vector<std::vector<long>> mat_;

        // per-trajectory state
        int maxTime_;
        int firstAfter_;
    };

    /**
     * @brief Histogram of each base parameter over `nbins` bins.
     */
    class DrawHistogramCollector final : public DataCollector {
    public:
        explicit DrawHistogramCollector(const std::vector<Parameter>& params, int nbins);
        DrawHistogramCollector(const DrawHistogramCollector& o);

        std::unique_ptr<DataCollector> clone() const override {
            return std::make_unique<DrawHistogramCollector>(*this);
        }

        void reset() override;
        void registerTime(double t) override {}
        void recordDraw(const Draw& draw) override;
        void merge(const DataCollector& other) override;
        void save() override;

        const std::vector<std::vector<long>>& histogram() const { return hist_; }

    private:
        std::vector<Parameter> params_;
        int nbins_;
        std::vector<std::vector<long>> hist_; // long-term
        int R0Bin_, rBin_, kBin_, alphaBin_, thetaBin_; // per-trajectory
    };

    /**
     * @brief 2D heatmap of (R0 vs. r) pairs.
     */
    class JointHeatmapCollector final : public DataCollector {
    public:
        JointHeatmapCollector(double R0min, double R0max, double rmin, double rmax, int bins);
        JointHeatmapCollector(const JointHeatmapCollector& o);

        std::unique_ptr<DataCollector> clone() const override {
            return std::make_unique<JointHeatmapCollector>(*this);
        }

        void reset() override;
        void registerTime(double t) override {}
        void recordDraw(const Draw& draw) override;
        void merge(const DataCollector& other) override;
        void save() override;


        const std::vector<std::vector<long>>& heatmap() const { return heat_; }

    private:
        double R0min_, R0max_, rmin_, rmax_;
        int bins_;
        std::vector<std::vector<long>> heat_;
        int binI_, binJ_; // per-trajectory
    };

    /**
     * @brief 1D histogram of either R0*r or alpha*theta.
     */
    class DerivedMarginalCollector final : public DataCollector {
    public:
        enum class Product { R0_r, AlphaTheta };

        DerivedMarginalCollector(Product prod, double lo, double hi, int bins);
        DerivedMarginalCollector(const DerivedMarginalCollector& o);

        std::unique_ptr<DataCollector> clone() const override {
            return std::make_unique<DerivedMarginalCollector>(*this);
        }

        void reset() override;
        void registerTime(double t) override {}
        void recordDraw(const Draw& draw) override;
        void merge(const DataCollector& other) override;
        void save() override;


        const std::vector<long>& histogram() const { return hist_; }

    private:
        Product prod_;
        double lo_, hi_;
        int bins_;
        std::vector<long> hist_;
        double val_; // per-trajectory
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

        void save() override {
            for (const auto& c : collectors_) c->save();
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
