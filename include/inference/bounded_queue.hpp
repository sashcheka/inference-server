#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace inference {

// A non-blocking producer interface makes overload handling an explicit choice
// for the caller. close() wakes consumers and lets them drain already queued work.
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0) {
            throw std::invalid_argument("bounded queue capacity must be greater than zero");
        }
    }

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    [[nodiscard]] bool try_push(T value) {
        {
            const std::lock_guard lock(mutex_);
            if (closed_ || items_.size() >= capacity_) {
                return false;
            }
            items_.push_back(std::move(value));
        }
        available_.notify_one();
        return true;
    }

    [[nodiscard]] std::optional<T> pop() {
        std::unique_lock lock(mutex_);
        available_.wait(lock, [this] { return closed_ || !items_.empty(); });

        if (items_.empty()) {
            return std::nullopt;
        }

        T item = std::move(items_.front());
        items_.pop_front();
        return item;
    }

    template <typename Rep, typename Period>
    [[nodiscard]] std::optional<T> pop_for(const std::chrono::duration<Rep, Period>& timeout) {
        return pop_until(std::chrono::steady_clock::now() + timeout);
    }

    [[nodiscard]] std::optional<T> pop_until(std::chrono::steady_clock::time_point deadline) {
        std::unique_lock lock(mutex_);
        if (!available_.wait_until(lock, deadline, [this] { return closed_ || !items_.empty(); })) {
            return std::nullopt;
        }
        if (items_.empty()) {
            return std::nullopt;
        }

        T item = std::move(items_.front());
        items_.pop_front();
        return item;
    }

    [[nodiscard]] std::optional<T> try_pop() {
        const std::lock_guard lock(mutex_);
        if (items_.empty()) {
            return std::nullopt;
        }

        T item = std::move(items_.front());
        items_.pop_front();
        return item;
    }

    void close() {
        {
            const std::lock_guard lock(mutex_);
            closed_ = true;
        }
        available_.notify_all();
    }

    [[nodiscard]] std::size_t size() const {
        const std::lock_guard lock(mutex_);
        return items_.size();
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] bool closed() const {
        const std::lock_guard lock(mutex_);
        return closed_;
    }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::deque<T> items_;
    bool closed_ = false;
};

}  // namespace inference
