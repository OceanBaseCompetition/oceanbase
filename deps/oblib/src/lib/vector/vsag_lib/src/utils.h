
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <queue>
#include <future>


#include "default_allocator.h"
#include "spdlog/spdlog.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"

namespace vsag {

template <typename T>
using UnorderedSet =
    std::unordered_set<T, std::hash<T>, std::equal_to<T>, vsag::AllocatorWrapper<T>>;

template <typename T>
using Vector = std::vector<T, vsag::AllocatorWrapper<T>>;

template <typename KeyType, typename ValType>
using UnorderedMap = std::unordered_map<KeyType,
                                        ValType,
                                        std::hash<KeyType>,
                                        std::equal_to<KeyType>,
                                        vsag::AllocatorWrapper<std::pair<const KeyType, ValType>>>;

struct SlowTaskTimer {
    SlowTaskTimer(const std::string& name, int64_t log_threshold_ms = 0);
    ~SlowTaskTimer();

    std::string name;
    int64_t threshold;
    std::chrono::steady_clock::time_point start;
};

struct Timer {
    Timer(double& ref);
    ~Timer();

    double& ref_;
    std::chrono::steady_clock::time_point start;
};

class WindowResultQueue {
public:
    WindowResultQueue();

    WindowResultQueue(size_t window_size);

    void
    Push(float value);

    size_t
    ResizeWindowSize(size_t new_window_size_);

    float
    GetAvgResult() const;

private:
    size_t count_ = 0;
    std::vector<float> queue_;
};

template <typename T>
struct Number {
    Number(T n) : num(n) {
    }

    bool
    in_range(T lower, T upper) {
        return ((unsigned)(num - lower) <= (upper - lower));
    }

    T num;
};

template <typename IndexOpParameters>
tl::expected<IndexOpParameters, Error>
try_parse_parameters(const std::string& json_string) {
    try {
        return IndexOpParameters::FromJson(json_string);
    } catch (const std::exception& e) {
        return tl::unexpected<Error>(ErrorType::INVALID_ARGUMENT, e.what());
    }
}

using tableint = unsigned int;
struct CompareByFirst {
    constexpr bool
    operator()(std::pair<float, tableint> const& a,
               std::pair<float, tableint> const& b) const noexcept {
        return a.first < b.first;
    }
};

// 维护优先队列的单例工作线程
class QueueWoker{
  public:
    static QueueWoker* getInstance(){
      // call_once是C++11确保只执行一次
      std::call_once(flag, []{instance_.store(new QueueWoker(), std::memory_order_release);});
      return instance_.load(std::memory_order_acquire);
    }

    // 唤醒该线程
    void wakeUp(std::priority_queue<std::pair<float, tableint>,
                        vsag::Vector<std::pair<float, tableint>>,
                        CompareByFirst>* queue) {
        p_queue_ = queue;
        is_sleeping_.store(false, std::memory_order_release);
        cv_.notify_one();
    }

    // 阻塞该线程
    void sleep() {
        is_sleeping_.store(true, std::memory_order_release);
    }

    // 提交 pop 操作
    void pop() {
        task_queue_.push(Task(TaskType::POP));
        std::atomic_thread_fence(std::memory_order_release);
    }

    // 提交 emplace 操作
    void emplace(const float& dist, const tableint& ep_id) {
        task_queue_.push(Task(TaskType::EMPLACE, dist, ep_id));
        std::atomic_thread_fence(std::memory_order_release);
    }

    // 同步执行 top 操作
    std::pair<float, tableint> top() {
        std::atomic_thread_fence(std::memory_order_acquire);
        while(!task_queue_.empty()){
            std::atomic_thread_fence(std::memory_order_acquire);
        }
        return p_queue_->top();
    }
    
    bool empty(){
        std::atomic_thread_fence(std::memory_order_acquire);
        while(!task_queue_.empty()){
            std::atomic_thread_fence(std::memory_order_acquire);
        }
        return p_queue_->empty();
    }

  private:
    QueueWoker(){
        worker_thread_ = std::thread(&QueueWoker::run, this);
    };
    QueueWoker(const QueueWoker&) = delete;
    QueueWoker& operator=(const QueueWoker&) = delete;
    enum class TaskType {
        EMPLACE,
        POP,
    };

    struct Task {
        TaskType type;
        float dist;    // 对于EMPLACE任务，dist参数
        tableint ep_id; // 对于EMPLACE任务，ep_id参数
        Task() = default;
        Task(TaskType t, float d = 0.0f, tableint id = 0)
            : type(t), dist(d), ep_id(id) {}
    };
    void run() {
        while (true) {
            std::atomic_thread_fence(std::memory_order_acquire);
            while (task_queue_.empty()){
                if(is_sleeping_.load(std::memory_order_acquire)){
                    std::unique_lock<std::mutex> lock(sleep_mutex_);
                    cv_.wait(lock, [this]() { return !is_sleeping_.load(std::memory_order_acquire);});
                }
                std::atomic_thread_fence(std::memory_order_acquire);
            }
            const Task& task = task_queue_.front();
            switch(task.type){
                case TaskType::EMPLACE:
                    p_queue_->emplace(task.dist, task.ep_id);
                    break;
                case TaskType::POP:
                    p_queue_->pop();
                    break;
            }
            task_queue_.pop();
            std::atomic_thread_fence(std::memory_order_release);
        }
    }
    std::queue<Task> task_queue_; 
    std::thread worker_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;

    std::atomic<bool> is_sleeping_{true};             // 是否处于睡眠状态
    std::mutex sleep_mutex_;
    std::priority_queue<std::pair<float, tableint>,
                        vsag::Vector<std::pair<float, tableint>>,
                        CompareByFirst>* p_queue_;

    static std::atomic<QueueWoker*> instance_;
    static std::once_flag flag;
};
}  // namespace vsag
