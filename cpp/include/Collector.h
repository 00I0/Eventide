#pragma once
/**
 * @file Collector.h
 * @brief Interfaces and implementations for collecting simulation data.
 */
#include <vector>

#include "CompiledExpression.h"
#include "Sampler.h"
#include "TrajectoryResult.h"

namespace eventide {
    /**
     * @brief Base interface for streaming & final data collection.
     */
    class DataCollector {
    public:
        virtual ~DataCollector() = default;

        /** @brief reset the per‑trajectory temporary state */
        virtual void reset() = 0;

        /**
         * @brief record a new infection event
         * @param parentInfectionTime  time of the parent infection
         * @param newInfectionTime     time of the new infection
         */
        virtual void registerTime(double parentInfectionTime, double newInfectionTime) = 0;

        /**
         * @brief record the draw parameters of an accepted trajectory
         * @param params   the Draw generating this trajectory
         */
        virtual void recordDraw(const Draw& params) = 0;

        /**
         * @brief commit per‑trajectory data into long‑term storage
         * @param trajectoryResult  trajectory outcome
         */
        virtual void save(TrajectoryResult trajectoryResult) = 0;

        /**
         * @brief merge another collector’s results into this one
         * @param other  same type collector to absorb
         */
        virtual void merge(const DataCollector& other) = 0;

        /** @brief clone a fresh, empty instance of this collector type */
        virtual std::unique_ptr<DataCollector> clone() const = 0;
    };

    /**
     * @brief (T+2)×(T+2) matrix of [floor(finalTime)][floor(firstAfterCutoff)] counts.
     */
    class TimeMatrixCollector final : public DataCollector {
    public:
        /**
         * @brief Construct a time‐matrix collector.
         * @param T           maximum day to tabulate
         * @param cutoffDay   day threshold for "first after"
         */
        explicit TimeMatrixCollector(int T, int cutoffDay);

        /**
         * @brief Copy constructor.
         * @param other the other TimeMatrixCollector to copy
         */
        TimeMatrixCollector(const TimeMatrixCollector& other);

        std::unique_ptr<DataCollector> clone() const override { return std::make_unique<TimeMatrixCollector>(*this); }

        void reset() override;
        void registerTime(double parentInfectionTime, double newInfectionTime) override;
        void recordDraw(const Draw& draw) override {}
        void merge(const DataCollector& other) override;
        void save(TrajectoryResult trajectoryResult) override;

        /**
         * @brief Access the accumulated matrix.
         * @return reference to a (T+2)x(T+2) count matrix
         */
        const std::vector<std::vector<long>>& matrix() const noexcept { return mat_; }

    private:
        int T_;
        int cutoffDay_;
        std::vector<std::vector<long>> mat_;

        // per-trajectory state
        int maxTime_;
        int firstAfter_;
    };


    /**
     * @brief One‐dimensional histogram of a compiled expression over Draws.
     */
    class Hist1D final : public DataCollector {
    public:
        /**
         * @brief Construct a 1D histogram collector.
         * @param expression  compiled expression mapping Draw→value
         * @param bins  number of bins
         * @param lo    lower bound
         * @param hi    upper bound
         */
        explicit Hist1D(const CompiledExpression& expression, int bins, double lo, double hi);

        /**
         * @brief Copy constructor.
         * @param other the other Hist1D to copy
         */
        Hist1D(const Hist1D& other);

        std::unique_ptr<DataCollector> clone() const override { return std::make_unique<Hist1D>(*this); }

        void reset() override {}
        void registerTime(const double parentInfectionTime, const double newInfectionTime) override {}
        void recordDraw(const Draw& draw) override;
        void merge(const DataCollector& other) override;
        void save(TrajectoryResult trajectoryResult) override;

        /**
         * @brief Retrieve the accumulated histogram counts.
         * @return vector of length `bins`
         */
        std::vector<long> histogram() const noexcept { return hist_; }

    private:
        CompiledExpression expression_;
        int bins_;
        double lo_, hi_;
        std::vector<long> hist_;
        double val_; // per trajectory
    };

    /**
     * @brief Two‐dimensional histogram of two compiled expressions over Draws.
     */
    class Hist2D final : public DataCollector {
    public:
        /**
         * @brief Construct a 2D histogram collector.
         * @param expressionX compiled expression for X
         * @param expressionY compiled expression for Y
         * @param bins  number of bins per dimension
         * @param loX   lower X bound
         * @param hiX   upper X bound
         * @param loY   lower Y bound
         * @param hiY   upper Y bound
         */
        explicit Hist2D(const CompiledExpression& expressionX, const CompiledExpression& expressionY,
                        int bins, double loX, double hiX, double loY, double hiY);

        /**
         * @brief Copy constructor.
         * @param other the other Hist2D to copy
         */
        Hist2D(const Hist2D& other);

        std::unique_ptr<DataCollector> clone() const override { return std::make_unique<Hist2D>(*this); }

        void reset() override {}
        void registerTime(const double parentInfectionTime, const double newInfectionTime) override {}
        void recordDraw(const Draw& draw) override;
        void merge(const DataCollector& other) override;
        void save(TrajectoryResult trajectoryResult) override;

        /**
         * @brief Retrieve the 2D histogram counts.
         * @return bins×bins matrix
         */
        std::vector<std::vector<long>> histogram() const noexcept { return hist_; }

    private:
        CompiledExpression expressionX_, expressionY_;
        int bins_;
        double loX_, hiX_, loY_, hiY_;
        std::vector<std::vector<long>> hist_;
        double valX_, valY_; // per trajectory
    };

    /**
     * @brief Collects the full parameter draws of each accepted trajectory.
     */
    class DrawCollector final : public DataCollector {
    public:
        explicit DrawCollector() = default;
        DrawCollector(const DrawCollector& other) = default;

        std::unique_ptr<DataCollector> clone() const override { return std::make_unique<DrawCollector>(*this); }

        void reset() override {}
        void registerTime(const double parentInfectionTime, const double newInfectionTime) override {}
        void recordDraw(const Draw& draw) override;
        void merge(const DataCollector& other) override;
        void save(TrajectoryResult trajectoryResult) override;

        /**
         * @brief Access collected draws.
         * @return vector of parameter‐tuples for each saved trajectory
         */
        std::vector<std::array<double, 5>> draws() const noexcept { return draws_; }

    private:
        std::vector<std::array<double, 5>> draws_;
        std::array<double, 5> currentDraw_;
    };


    /**
     * @brief Collects the size of active infection set at specified time points.
     */
    class ActiveSetSizeCollector final : public DataCollector {
    public:
        /**
         * @brief Construct an active set size collector.
         * @param collectionTime time point at which to measure active set size
         */
        explicit ActiveSetSizeCollector(double collectionTime);

        /**
         * @brief Copy constructor.
         * @param other the other ActiveSetSizeCollector to copy
         */
        ActiveSetSizeCollector(const ActiveSetSizeCollector& other) = default;

        std::unique_ptr<DataCollector> clone() const override {
            return std::make_unique<ActiveSetSizeCollector>(*this);
        }

        void reset() override;
        void registerTime(double parentInfectionTime, double newInfectionTime) override;
        void recordDraw(const Draw& draw) override {}
        void merge(const DataCollector& other) override;
        void save(TrajectoryResult trajectoryResult) override;

        /**
         * @brief Access collected active set sizes.
         * @return vector of active set sizes at collection time points
         */
        std::vector<double> activeSetSizes() const noexcept { return activeSetSizes_; }

    private:
        int currentActiveSetSize = 0;
        double collectionTime_;
        std::vector<double> activeSetSizes_;
    };


    /**
     * @brief Collects all infection event times for each accepted trajectory.
     */
    class InfectionTimeCollector final : public DataCollector {
    public:
        explicit InfectionTimeCollector() = default;

        /**
         * @brief Copy constructor.
         * @param other the other InfectionTimeCollector to copy
         */
        InfectionTimeCollector(const InfectionTimeCollector& other) = default;

        std::unique_ptr<DataCollector> clone() const override {
            return std::make_unique<InfectionTimeCollector>(*this);
        }

        void reset() override;
        void registerTime(double parentInfectionTime, double newInfectionTime) override;
        void recordDraw(const Draw& draw) override {}
        void merge(const DataCollector& other) override;
        void save(TrajectoryResult trajectoryResult) override;

        /**
         * @brief Access collected infection times.
         * @return vector of infection time vectors, one per saved trajectory
         */
        std::vector<std::vector<double>> infectionTimes() const noexcept { return allInfectionTimes_; }

    private:
        std::vector<double> currentInfectionTimes_;
        std::vector<std::vector<double>> allInfectionTimes_;
    };

    /**
     * @brief Thread‑local grouping of multiple DataCollector instances.
     *
     * Internally owns unique_ptr<DataCollector> clones.
     */
    class DataCollectorGroup final : public DataCollector {
    public:
        /**
         * @brief Construct by cloning each supplied DataCollector.
         * @param collectors  original collectors to clone
         */
        explicit DataCollectorGroup(const std::vector<std::unique_ptr<DataCollector>>& collectors);

        /**
         * @brief Construct by cloning from shared_ptr collectors.
         * @param collectors  original collectors to clone
         */
        explicit DataCollectorGroup(const std::vector<std::shared_ptr<DataCollector>>& collectors);


        /**
         * @brief Copy constructor.
         * @param other the other DataCollectorGroup to copy
         */
        DataCollectorGroup(const DataCollectorGroup& other);

        std::unique_ptr<DataCollector> clone() const override { return std::make_unique<DataCollectorGroup>(*this); }

        void reset() override {
            for (const auto& c : collectors_) c->reset();
        }

        void registerTime(const double parentInfectionTime, const double newInfectionTime) override {
            for (const auto& c : collectors_) c->registerTime(parentInfectionTime, newInfectionTime);
        }

        void recordDraw(const Draw& params) override {
            for (const auto& c : collectors_) c->recordDraw(params);
        }

        void merge(const DataCollector& other) override;

        void save(const TrajectoryResult trajectoryResult) override {
            for (const auto& c : collectors_) c->save(trajectoryResult);
        }

        /** @brief Number of collectors in this group. */
        size_t size() const { return collectors_.size(); }

        /**
         * @brief Access a specific collector.
         * @param i  index [0...size())
         */
        const std::unique_ptr<DataCollector>& at(const size_t i) const { return collectors_.at(i); }

    private:
        std::vector<std::unique_ptr<DataCollector>> collectors_;
    };
}
