
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

// 计算距离的单例工作线程
class CalWoker{
  public:
    static CalWoker* getInstance(){
      // call_once是C++11确保只执行一次
      std::call_once(flag, []{instance_.store(new CalWoker(), std::memory_order_release);});
      return instance_.load(std::memory_order_acquire);
    }

    // 提交任务
    void submitTask(const std::function<void()>& task) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            task_queue_.emplace(task);
        }
        cv_.notify_one();
    }

  private:
    CalWoker(){
      worker_thread_ = std::thread(&CalWoker::run, this);
    };
    CalWoker(const CalWoker&) = delete;
    CalWoker& operator=(const CalWoker&) = delete;

    void run() {
      while (true) {
          std::function<void()> task;
          {
              std::unique_lock<std::mutex> lock(mutex_);
              cv_.wait(lock, [this]() { return !task_queue_.empty(); });
              task = std::move(task_queue_.front());
              task_queue_.pop();
          }
          task(); // 执行任务
      }
    }
    std::queue<std::function<void()>> task_queue_;
    std::thread worker_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;

    static std::atomic<CalWoker*> instance_;
    static std::once_flag flag;
};
}  // namespace vsag
