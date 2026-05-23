#pragma once
#include <vector>
#include <cstddef>
#include <cassert>
#include <utility>

namespace eventide {
    /**
     * Simple d-ary min-heap specialized for double values.
     *
     * Template parameter:
     *   D = arity (>=2). Default: 2 (binary heap).
     *
     * Behaviour:
     *   - top() returns the smallest element.
     *   - No thread-safety (intended per-worker).
     *   - No allocations after reserve_space().
     */
    template <typename T = double, std::size_t D = 2>
    class DaryHeap {
        static_assert(D >= 2, "D must be >= 2");

    public:
        using size_type = std::size_t;

        DaryHeap() = default;
        explicit DaryHeap(const size_type reserve) { reserve_space(reserve); }

        void reserve_space(const size_type n) { data_.reserve(n); }
        void clear() noexcept { data_.clear(); }
        bool empty() const noexcept { return data_.empty(); }
        size_type size() const noexcept { return data_.size(); }

        const T& top() const {
            assert(!data_.empty());
            return data_.front();
        }

        void push(const T& v) {
            data_.push_back(v);
            sift_up(data_.size() - 1);
        }

        // pop the smallest element (no return)
        void pop() {
            assert(!data_.empty());
            if (const size_type n = data_.size(); n == 1) {
                data_.pop_back();
                return;
            }
            data_[0] = std::move(data_.back());
            data_.pop_back();
            sift_down(0);
        }

        // pop and return the smallest element
        T pop_top() {
            assert(!data_.empty());
            const T out = std::move(data_.front());
            pop();
            return out;
        }

    private:
        std::vector<T> data_;

        static constexpr size_type parent_of(const size_type i) noexcept { return (i - 1) / D; }

        static constexpr size_type first_child_of(const size_type i) noexcept { return i * D + 1; }

        void sift_up(size_type idx) {
            while (idx > 0) {
                if (const size_type p = parent_of(idx); data_[p] > data_[idx]) {
                    std::swap(data_[p], data_[idx]);
                    idx = p;
                }
                else {
                    break;
                }
            }
        }

        void sift_down(size_type idx) {
            const size_type n = data_.size();
            while (true) {
                size_type best = idx;
                const size_type firstChild = first_child_of(idx);
                for (size_type k = 0; k < D; ++k) {
                    const size_type c = firstChild + k;
                    if (c >= n) break;
                    if (data_[c] < data_[best]) best = c;
                }
                if (best == idx) break;
                std::swap(data_[idx], data_[best]);
                idx = best;
            }
        }
    };
} // namespace eventide
