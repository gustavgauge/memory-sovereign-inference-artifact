#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace msi::bounded_source {

using Clock = std::chrono::steady_clock;

struct Extent {
  std::uint64_t source_offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t destination_offset = 0;
};

struct Window {
  std::uint64_t window_id = 0;
  std::uint64_t address = 0;
  std::uint64_t bytes = 0;
  std::uint64_t direct_scratch_address = 0;
  std::uint64_t direct_scratch_bytes = 0;
};

struct Ticket {
  std::uint64_t ticket_id = 0;
  std::uint64_t window_id = 0;
  std::uint64_t generation = 0;
};

enum class FaultMode : std::uint8_t {
  None,
  ShortSuccess,
  IoError,
  PartialBundle,
};

struct Telemetry {
  std::uint64_t submissions = 0;
  std::uint64_t completions = 0;
  std::uint64_t cancellations = 0;
  std::uint64_t failures = 0;
  std::uint64_t logical_bytes = 0;
  std::uint64_t physical_read_bytes = 0;
  std::uint64_t block_read_bytes = 0;
  std::uint64_t padding_bytes = 0;
  std::uint64_t read_wall_ns = 0;
  std::uint64_t exposed_wait_ns = 0;
  std::uint64_t h2d_issued_bytes = 0;
  std::uint64_t h2d_completed_bytes = 0;
  std::uint64_t queue_rejections = 0;
  std::uint64_t lifecycle_rejections = 0;
  std::uint64_t generation_reuses = 0;
  std::uint64_t fixed_direct_reads = 0;
  std::uint64_t dynamic_direct_allocations = 0;
  std::uint64_t injected_short_completions = 0;
  std::uint64_t injected_io_errors = 0;
  std::uint64_t injected_partial_bundles = 0;
  std::uint64_t injected_completed_extents = 0;
  std::uint64_t current_in_flight = 0;
  std::uint64_t peak_in_flight = 0;
  std::uint64_t active_tickets = 0;
  std::uint64_t free_windows = 0;
  std::uint64_t filling_windows = 0;
  std::uint64_t ready_windows = 0;
  std::uint64_t copying_windows = 0;
  std::uint64_t retirable_windows = 0;
};

class Service {
 public:
  Service(std::string path, std::vector<Window> windows,
          std::uint64_t max_in_flight, bool direct_io,
          std::uint64_t direct_alignment = 4096,
          bool discard_buffered_cache = false,
          FaultMode fault_mode = FaultMode::None)
      : path_(std::move(path)),
        max_in_flight_(max_in_flight),
        direct_io_(direct_io),
        direct_alignment_(direct_alignment),
        discard_buffered_cache_(discard_buffered_cache),
        fault_mode_(fault_mode) {
    if (path_.empty() || windows.empty() || max_in_flight_ == 0 ||
        max_in_flight_ > windows.size() || direct_alignment_ == 0 ||
        (direct_alignment_ & (direct_alignment_ - 1)) != 0) {
      throw std::invalid_argument("invalid bounded source-service geometry");
    }
    struct stat info {};
    if (::stat(path_.c_str(), &info) != 0 || info.st_size <= 0) {
      throw std::runtime_error("bounded source stat failed: " +
                               std::string(std::strerror(errno)));
    }
    source_bytes_ = static_cast<std::uint64_t>(info.st_size);
    for (const Window& window : windows) {
      if (window.window_id == 0 || window.address == 0 || window.bytes == 0 ||
          !windows_.emplace(window.window_id, WindowState{window}).second) {
        throw std::invalid_argument("invalid bounded source-service window");
      }
    }
    int flags = O_RDONLY;
#ifdef O_DIRECT
    if (direct_io_) {
      flags |= O_DIRECT;
    }
#else
    if (direct_io_) {
      throw std::runtime_error("O_DIRECT is unavailable on this platform");
    }
#endif
    fd_ = ::open(path_.c_str(), flags);
    if (fd_ < 0) {
      throw std::runtime_error("bounded source open failed: " +
                               std::string(std::strerror(errno)));
    }
  }

  Service(const Service&) = delete;
  Service& operator=(const Service&) = delete;

  ~Service() { close_noexcept(); }

  Ticket submit(std::uint64_t window_id, std::vector<Extent> extents) {
    validate_extents(window_id, extents);
    auto task = std::make_shared<Task>();
    Ticket ticket;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      require_open_locked();
      if (in_flight_ >= max_in_flight_) {
        ++telemetry_.queue_rejections;
        throw std::runtime_error("bounded source queue is full");
      }
      WindowState& window = find_window_locked(window_id);
      if (window.state != State::Free) {
        ++telemetry_.lifecycle_rejections;
        throw std::logic_error("bounded source window is not free");
      }
      if (window.generation != 0) {
        ++telemetry_.generation_reuses;
      }
      ticket = Ticket{++next_ticket_id_, window_id, ++window.generation};
      task->ticket = ticket;
      task->extents = std::move(extents);
      task->window = window.window;
      window.state = State::Filling;
      window.ticket_id = ticket.ticket_id;
      tasks_.emplace(ticket.ticket_id, task);
      ++in_flight_;
      ++telemetry_.submissions;
      telemetry_.peak_in_flight = std::max(telemetry_.peak_in_flight, in_flight_);
    }
    task->future = std::async(std::launch::async, [this, task] {
      return execute(*task);
    }).share();
    return ticket;
  }

  void await(const Ticket& ticket) {
    const auto wait_begin = Clock::now();
    std::shared_ptr<Task> task = find_task(ticket);
    Outcome outcome;
    try {
      outcome = task->future.get();
    } catch (...) {
      finish_failed(ticket);
      throw;
    }
    const std::uint64_t wait_ns = elapsed_ns(wait_begin, Clock::now());
    std::lock_guard<std::mutex> lock(mutex_);
    WindowState& window = validate_ticket_locked(ticket);
    if (window.state != State::Filling || in_flight_ == 0) {
      ++telemetry_.lifecycle_rejections;
      throw std::logic_error("bounded source completion state mismatch");
    }
    --in_flight_;
    telemetry_.exposed_wait_ns += wait_ns;
    telemetry_.read_wall_ns += outcome.read_wall_ns;
    telemetry_.logical_bytes += outcome.logical_bytes;
    telemetry_.physical_read_bytes += outcome.physical_bytes;
    telemetry_.block_read_bytes += outcome.block_bytes;
    telemetry_.padding_bytes += outcome.padding_bytes;
    telemetry_.fixed_direct_reads += outcome.fixed_direct_reads;
    telemetry_.dynamic_direct_allocations +=
        outcome.dynamic_direct_allocations;
    if (outcome.fault != FaultMode::None) {
      telemetry_.injected_completed_extents += outcome.completed_extents;
      window.state = State::Retirable;
      ++telemetry_.failures;
      if (outcome.fault == FaultMode::ShortSuccess) {
        ++telemetry_.injected_short_completions;
        throw std::runtime_error("bounded source injected short successful completion");
      }
      if (outcome.fault == FaultMode::IoError) {
        ++telemetry_.injected_io_errors;
        throw std::runtime_error("bounded source injected EIO completion");
      }
      ++telemetry_.injected_partial_bundles;
      throw std::runtime_error("bounded source injected partial-bundle failure");
    }
    if (outcome.cancelled) {
      window.state = State::Retirable;
      ++telemetry_.cancellations;
    } else {
      window.state = State::Ready;
      ++telemetry_.completions;
    }
  }

  void cancel(const Ticket& ticket) {
    std::shared_ptr<Task> task = find_task(ticket);
    task->cancel_requested.store(true);
    await(ticket);
  }

  void begin_h2d(const Ticket& ticket, std::uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    WindowState& window = validate_ticket_locked(ticket);
    if (window.state != State::Ready || bytes == 0 || bytes > window.window.bytes) {
      ++telemetry_.lifecycle_rejections;
      throw std::logic_error("invalid bounded source H2D begin");
    }
    window.state = State::Copying;
    window.h2d_bytes = bytes;
    telemetry_.h2d_issued_bytes += bytes;
  }

  void complete_h2d(const Ticket& ticket, std::uint64_t bytes,
                    std::uint64_t completion_event) {
    std::lock_guard<std::mutex> lock(mutex_);
    WindowState& window = validate_ticket_locked(ticket);
    if (window.state != State::Copying || bytes != window.h2d_bytes ||
        completion_event == 0) {
      ++telemetry_.lifecycle_rejections;
      throw std::logic_error("invalid bounded source H2D completion");
    }
    window.state = State::Retirable;
    window.completion_event = completion_event;
    telemetry_.h2d_completed_bytes += bytes;
  }

  void mark_retirable(const Ticket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    WindowState& window = validate_ticket_locked(ticket);
    if (window.state != State::Ready) {
      ++telemetry_.lifecycle_rejections;
      throw std::logic_error("bounded source window is not ready");
    }
    window.state = State::Retirable;
  }

  void retire(const Ticket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    WindowState& window = validate_ticket_locked(ticket);
    if (window.state != State::Retirable) {
      ++telemetry_.lifecycle_rejections;
      throw std::logic_error("bounded source window is not retirable");
    }
    window.state = State::Free;
    window.ticket_id = 0;
    window.h2d_bytes = 0;
    window.completion_event = 0;
    tasks_.erase(ticket.ticket_id);
  }

  Telemetry snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Telemetry result = telemetry_;
    result.current_in_flight = in_flight_;
    result.active_tickets = tasks_.size();
    for (const auto& [id, window] : windows_) {
      (void)id;
      switch (window.state) {
        case State::Free: ++result.free_windows; break;
        case State::Filling: ++result.filling_windows; break;
        case State::Ready: ++result.ready_windows; break;
        case State::Copying: ++result.copying_windows; break;
        case State::Retirable: ++result.retirable_windows; break;
      }
    }
    return result;
  }

  std::uint64_t source_bytes() const { return source_bytes_; }
  std::uint64_t max_in_flight() const { return max_in_flight_; }
  bool direct_io() const { return direct_io_; }

  void shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }
    if (in_flight_ != 0 || !tasks_.empty() ||
        std::any_of(windows_.begin(), windows_.end(), [](const auto& row) {
          return row.second.state != State::Free;
        })) {
      ++telemetry_.lifecycle_rejections;
      throw std::logic_error("bounded source shutdown with live state");
    }
    ::close(fd_);
    fd_ = -1;
    closed_ = true;
  }

 private:
  enum class State { Free, Filling, Ready, Copying, Retirable };

  struct WindowState {
    explicit WindowState(Window value) : window(value) {}
    Window window;
    State state = State::Free;
    std::uint64_t generation = 0;
    std::uint64_t ticket_id = 0;
    std::uint64_t h2d_bytes = 0;
    std::uint64_t completion_event = 0;
  };

  struct Outcome {
    std::uint64_t logical_bytes = 0;
    std::uint64_t physical_bytes = 0;
    std::uint64_t block_bytes = 0;
    std::uint64_t padding_bytes = 0;
    std::uint64_t read_wall_ns = 0;
    std::uint64_t fixed_direct_reads = 0;
    std::uint64_t dynamic_direct_allocations = 0;
    std::uint64_t completed_extents = 0;
    FaultMode fault = FaultMode::None;
    bool cancelled = false;
  };

  struct Task {
    Ticket ticket;
    Window window;
    std::vector<Extent> extents;
    std::atomic<bool> cancel_requested{false};
    std::shared_future<Outcome> future;
  };

  static std::uint64_t elapsed_ns(Clock::time_point begin,
                                  Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
  }

  static std::uint64_t round_down(std::uint64_t value,
                                  std::uint64_t alignment) {
    return value & ~(alignment - 1);
  }

  static std::uint64_t round_up(std::uint64_t value,
                                std::uint64_t alignment) {
    if (value > UINT64_MAX - (alignment - 1)) {
      throw std::overflow_error("bounded source alignment overflow");
    }
    return (value + alignment - 1) & ~(alignment - 1);
  }

  void validate_extents(std::uint64_t window_id,
                        const std::vector<Extent>& extents) const {
    const auto window = windows_.find(window_id);
    if (window == windows_.end() || extents.empty()) {
      throw std::invalid_argument("unknown bounded source window or empty request");
    }
    for (const Extent& extent : extents) {
      if (extent.bytes == 0 || extent.source_offset > source_bytes_ ||
          extent.bytes > source_bytes_ - extent.source_offset ||
          extent.destination_offset > window->second.window.bytes ||
          extent.bytes > window->second.window.bytes - extent.destination_offset) {
        throw std::invalid_argument("bounded source extent is out of bounds");
      }
    }
  }

  Outcome execute(const Task& task) const {
    const auto begin = Clock::now();
    Outcome result;
    const bool inject = fault_mode_ != FaultMode::None &&
                        !fault_claimed_.exchange(true);
    if (inject && fault_mode_ == FaultMode::IoError) {
      result.fault = FaultMode::IoError;
      result.read_wall_ns = elapsed_ns(begin, Clock::now());
      return result;
    }
    for (std::size_t extent_index = 0; extent_index < task.extents.size();
         ++extent_index) {
      const Extent& extent = task.extents[extent_index];
      if (task.cancel_requested.load()) {
        result.cancelled = true;
        break;
      }
      if (inject && fault_mode_ == FaultMode::ShortSuccess && extent_index == 0) {
        result.logical_bytes += extent.bytes - 1;
        result.fault = FaultMode::ShortSuccess;
        result.read_wall_ns = elapsed_ns(begin, Clock::now());
        return result;
      }
      void* destination = reinterpret_cast<void*>(
          static_cast<std::uintptr_t>(task.window.address +
                                      extent.destination_offset));
      if (!direct_io_) {
        read_exact(destination, extent.bytes, extent.source_offset);
        if (discard_buffered_cache_) {
          const int status = ::posix_fadvise(fd_,
              static_cast<off_t>(extent.source_offset),
              static_cast<off_t>(extent.bytes), POSIX_FADV_DONTNEED);
          if (status != 0) {
            throw std::runtime_error("bounded source cache discard failed: " +
                                     std::string(std::strerror(status)));
          }
        }
        result.logical_bytes += extent.bytes;
        ++result.completed_extents;
        if (inject && fault_mode_ == FaultMode::PartialBundle && extent_index == 0) {
          result.fault = FaultMode::PartialBundle;
          result.read_wall_ns = elapsed_ns(begin, Clock::now());
          return result;
        }
        continue;
      }
      const bool direct_destination =
          extent.source_offset % direct_alignment_ == 0 &&
          extent.bytes % direct_alignment_ == 0 &&
          reinterpret_cast<std::uintptr_t>(destination) % direct_alignment_ == 0;
      if (direct_destination) {
        read_exact(destination, extent.bytes, extent.source_offset);
        result.logical_bytes += extent.bytes;
        result.physical_bytes += extent.bytes;
        result.block_bytes += extent.bytes;
        ++result.completed_extents;
        if (inject && fault_mode_ == FaultMode::PartialBundle && extent_index == 0) {
          result.fault = FaultMode::PartialBundle;
          result.read_wall_ns = elapsed_ns(begin, Clock::now());
          return result;
        }
        continue;
      }
      const std::uint64_t aligned_begin =
          round_down(extent.source_offset, direct_alignment_);
      const std::uint64_t aligned_end = round_up(
          extent.source_offset + extent.bytes, direct_alignment_);
      const std::uint64_t envelope_bytes = aligned_end - aligned_begin;
      void* envelope = nullptr;
      bool dynamic_envelope = false;
      if (task.window.direct_scratch_address != 0 &&
          task.window.direct_scratch_bytes >= envelope_bytes &&
          task.window.direct_scratch_address % direct_alignment_ == 0) {
        envelope = reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(task.window.direct_scratch_address));
        ++result.fixed_direct_reads;
      } else {
        const int allocation = ::posix_memalign(
            &envelope, static_cast<std::size_t>(direct_alignment_),
            static_cast<std::size_t>(envelope_bytes));
        if (allocation != 0 || envelope == nullptr) {
          throw std::bad_alloc();
        }
        dynamic_envelope = true;
        ++result.dynamic_direct_allocations;
      }
      try {
        read_exact(envelope, envelope_bytes, aligned_begin);
        std::memcpy(destination,
                    static_cast<const char*>(envelope) +
                        (extent.source_offset - aligned_begin),
                    static_cast<std::size_t>(extent.bytes));
      } catch (...) {
        if (dynamic_envelope) {
          ::free(envelope);
        }
        throw;
      }
      if (dynamic_envelope) {
        ::free(envelope);
      }
      result.logical_bytes += extent.bytes;
      result.physical_bytes += envelope_bytes;
      result.block_bytes += envelope_bytes;
      result.padding_bytes += envelope_bytes - extent.bytes;
      ++result.completed_extents;
      if (inject && fault_mode_ == FaultMode::PartialBundle && extent_index == 0) {
        result.fault = FaultMode::PartialBundle;
        result.read_wall_ns = elapsed_ns(begin, Clock::now());
        return result;
      }
    }
    result.read_wall_ns = elapsed_ns(begin, Clock::now());
    return result;
  }

  void read_exact(void* destination, std::uint64_t bytes,
                  std::uint64_t offset) const {
    std::uint64_t complete = 0;
    while (complete < bytes) {
      const ssize_t count = ::pread(
          fd_, static_cast<char*>(destination) + complete,
          static_cast<std::size_t>(bytes - complete),
          static_cast<off_t>(offset + complete));
      if (count < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::runtime_error("bounded source pread failed: " +
                                 std::string(std::strerror(errno)));
      }
      if (count == 0) {
        throw std::runtime_error("bounded source encountered unexpected EOF");
      }
      complete += static_cast<std::uint64_t>(count);
    }
  }

  std::shared_ptr<Task> find_task(const Ticket& ticket) const {
    std::lock_guard<std::mutex> lock(mutex_);
    validate_ticket_locked(ticket);
    const auto task = tasks_.find(ticket.ticket_id);
    if (task == tasks_.end()) {
      throw std::logic_error("unknown bounded source ticket");
    }
    return task->second;
  }

  WindowState& find_window_locked(std::uint64_t window_id) {
    const auto window = windows_.find(window_id);
    if (window == windows_.end()) {
      throw std::invalid_argument("unknown bounded source window");
    }
    return window->second;
  }

  WindowState& validate_ticket_locked(const Ticket& ticket) const {
    const auto iterator = windows_.find(ticket.window_id);
    if (iterator == windows_.end() ||
        iterator->second.ticket_id != ticket.ticket_id ||
        iterator->second.generation != ticket.generation) {
      throw std::logic_error("stale bounded source ticket");
    }
    return const_cast<WindowState&>(iterator->second);
  }

  void require_open_locked() const {
    if (closed_ || fd_ < 0) {
      throw std::logic_error("bounded source service is closed");
    }
  }

  void finish_failed(const Ticket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    WindowState& window = validate_ticket_locked(ticket);
    if (window.state == State::Filling && in_flight_ != 0) {
      --in_flight_;
    }
    window.state = State::Retirable;
    ++telemetry_.failures;
  }

  void close_noexcept() noexcept {
    std::vector<std::shared_ptr<Task>> tasks;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& [id, task] : tasks_) {
        (void)id;
        task->cancel_requested.store(true);
        tasks.push_back(task);
      }
    }
    for (const auto& task : tasks) {
      try {
        task->future.wait();
      } catch (...) {
      }
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    closed_ = true;
  }

  std::string path_;
  std::uint64_t source_bytes_ = 0;
  std::uint64_t max_in_flight_ = 0;
  bool direct_io_ = false;
  std::uint64_t direct_alignment_ = 4096;
  bool discard_buffered_cache_ = false;
  FaultMode fault_mode_ = FaultMode::None;
  mutable std::atomic<bool> fault_claimed_{false};
  int fd_ = -1;
  mutable std::mutex mutex_;
  bool closed_ = false;
  std::uint64_t next_ticket_id_ = 0;
  std::uint64_t in_flight_ = 0;
  Telemetry telemetry_;
  std::unordered_map<std::uint64_t, WindowState> windows_;
  std::unordered_map<std::uint64_t, std::shared_ptr<Task>> tasks_;
};

}  // namespace msi::bounded_source
