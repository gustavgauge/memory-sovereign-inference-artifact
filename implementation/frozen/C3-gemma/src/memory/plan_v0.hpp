#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace msi::plan_v0 {

using Id = std::uint64_t;
constexpr Id kInvalidId = std::numeric_limits<Id>::max();
constexpr std::uint32_t kInvalidConsumer =
    std::numeric_limits<std::uint32_t>::max();

enum class Lifecycle : std::uint8_t {
  constructed,
  ready,
  request_active,
  shutdown,
};

enum class HostWindowState : std::uint8_t {
  free,
  filling,
  ready,
  copying,
  recyclable,
};

struct SourceExtent {
  Id source_id = kInvalidId;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  Id extent_identity = kInvalidId;
};

struct ArenaManifest {
  Id arena_id = kInvalidId;
  std::uint64_t stable_address = 0;
  std::uint64_t bytes = 0;
};

struct ObjectManifest {
  Id object_id = kInvalidId;
  Id tensor_id = kInvalidId;
  Id graph_role_id = kInvalidId;
  std::uint64_t stable_address = 0;
  SourceExtent source;
  std::uint32_t max_consumers = 1;
  Id host_window_id = kInvalidId;
  std::vector<Id> allowed_host_window_ids;
  Id gpu_slot_id = kInvalidId;
  std::vector<Id> allowed_gpu_slot_ids;
};

struct SlotManifest {
  Id slot_id = kInvalidId;
  std::uint64_t bytes = 0;
  std::uint64_t stable_address = 0;
};

struct HostWindowManifest {
  Id window_id = kInvalidId;
  std::uint64_t stable_address = 0;
  std::uint64_t bytes = 0;
};

struct Manifest {
  Id manifest_identity = kInvalidId;
  ArenaManifest arena;
  std::vector<ObjectManifest> objects;
  std::vector<SlotManifest> gpu_slots;
  std::vector<HostWindowManifest> host_windows;
  std::uint32_t max_state_transactions = 0;
};

struct Ticket {
  Id ticket_id = kInvalidId;
  Id request_epoch = kInvalidId;
  Id object_id = kInvalidId;
  Id slot_id = kInvalidId;
  Id generation = 0;
  std::uint32_t consumer = 0;
  Id window_id = kInvalidId;
  Id window_generation = 0;
};

struct WindowTicket {
  Id request_epoch = kInvalidId;
  Id object_id = kInvalidId;
  Id window_id = kInvalidId;
  Id generation = 0;
};

struct CopyTicket {
  Id request_epoch = kInvalidId;
  Id object_id = kInvalidId;
  Id window_id = kInvalidId;
  Id window_generation = 0;
  Id slot_id = kInvalidId;
  Id slot_generation = 0;
  std::uint32_t consumer = 0;
};

struct StateTicket {
  Id transaction_id = kInvalidId;
  Id request_epoch = kInvalidId;
  Id branch_id = kInvalidId;
  Id parent_generation = 0;
  Id candidate_generation = 0;
};

struct CostTerms {
  double output_tokens = 0.0;
  double commitment = 0.0;
  double prompt_ms = 0.0;
  double proposer_ms = 0.0;
  double target_compute_ms = 0.0;
  double storage_ms = 0.0;
  double h2d_ms = 0.0;
  bool overlap_service = true;

  double request_ms() const {
    if (output_tokens <= 0.0 || commitment <= 0.0) {
      return 0.0;
    }
    const double service_ms = overlap_service
        ? std::max({target_compute_ms, storage_ms, h2d_ms})
        : target_compute_ms + storage_ms + h2d_ms;
    return prompt_ms + (output_tokens / commitment) * (proposer_ms + service_ms);
  }

  double tokens_per_second() const {
    const double milliseconds = request_ms();
    return milliseconds > 0.0 ? output_tokens * 1000.0 / milliseconds : 0.0;
  }
};

struct Telemetry {
  std::uint64_t requests_begun = 0;
  std::uint64_t requests_finished = 0;
  std::uint64_t bindings = 0;
  std::uint64_t slot_releases = 0;
  std::uint64_t readiness_events = 0;
  std::uint64_t consumer_acquires = 0;
  std::uint64_t consumer_completions = 0;
  std::uint64_t state_stages = 0;
  std::uint64_t state_commits = 0;
  std::uint64_t state_rollbacks = 0;
  std::uint64_t resets = 0;
  std::uint64_t shutdowns = 0;
  std::uint64_t storage_bytes = 0;
  std::uint64_t h2d_bytes = 0;
  std::uint64_t d2h_bytes = 0;
  std::uint64_t scheduled_objects = 0;
  std::uint64_t scheduled_source_bytes = 0;
  std::uint64_t source_reads_issued = 0;
  std::uint64_t source_reads_completed = 0;
  std::uint64_t completed_application_read_bytes = 0;
  std::uint64_t h2d_issued_bytes = 0;
  std::uint64_t h2d_completed_bytes = 0;
  std::uint64_t consumed_source_objects = 0;
  std::uint64_t consumer_used_source_bytes = 0;
  std::uint64_t migration_reused_objects = 0;
  std::uint64_t migration_reused_bytes = 0;
  std::uint64_t unused_objects = 0;
  std::uint64_t speculative_unused_bytes = 0;
  std::uint64_t window_recycles = 0;
  std::uint64_t exposed_source_wait_ns = 0;
  std::uint64_t exposed_h2d_wait_ns = 0;
  std::uint64_t request_runtime_object_visits = 0;
  std::uint64_t cost_records = 0;
  std::uint64_t missing_ready_rejections = 0;
  std::uint64_t stale_generation_rejections = 0;
  std::uint64_t wrong_object_rejections = 0;
  std::uint64_t premature_reuse_rejections = 0;
  std::uint64_t missing_completion_rejections = 0;
  std::uint64_t duplicate_completion_rejections = 0;
  std::uint64_t lifetime_rejections = 0;
  std::uint64_t lifecycle_rejections = 0;
  std::uint64_t capacity_rejections = 0;
  std::uint64_t state_rejections = 0;
};

struct Snapshot {
  Lifecycle lifecycle = Lifecycle::constructed;
  Id manifest_identity = kInvalidId;
  Id request_id = kInvalidId;
  Id request_epoch = 0;
  Id committed_state_generation = 0;
  std::uint64_t ready_slots = 0;
  std::uint64_t bound_slots = 0;
  std::uint64_t live_consumers = 0;
  std::uint64_t staged_states = 0;
  std::uint64_t free_windows = 0;
  std::uint64_t filling_windows = 0;
  std::uint64_t ready_windows = 0;
  std::uint64_t copying_windows = 0;
  std::uint64_t recyclable_windows = 0;
  Telemetry telemetry;
  CostTerms last_cost;
};

class Engine {
 public:
  explicit Engine(Manifest manifest) : manifest_(std::move(manifest)) {
    validate_manifest();
    object_runtime_.reserve(manifest_.objects.size());
    for (const auto& object : manifest_.objects) {
      ObjectRuntime runtime;
      runtime.live_tickets.assign(object.max_consumers, kInvalidId);
      runtime.completed.assign(object.max_consumers, false);
      object_index_.emplace(object.object_id, object_runtime_.size());
      object_runtime_.push_back(std::move(runtime));
    }
    object_to_slot_.assign(object_runtime_.size(), kInvalidIndex);
    scheduled_object_indices_.reserve(object_runtime_.size());
    slot_runtime_.resize(manifest_.gpu_slots.size());
    for (std::size_t index = 0; index < manifest_.gpu_slots.size(); ++index) {
      slot_index_.emplace(manifest_.gpu_slots[index].slot_id, index);
    }
    window_runtime_.resize(manifest_.host_windows.size());
    for (std::size_t index = 0; index < manifest_.host_windows.size(); ++index) {
      window_index_.emplace(manifest_.host_windows[index].window_id, index);
    }
    state_runtime_.resize(manifest_.max_state_transactions);
  }

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  void initialize(Id committed_state_generation = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::constructed, "initialize");
    committed_state_generation_ = committed_state_generation;
    lifecycle_ = Lifecycle::ready;
  }

  void begin_request(Id request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::ready, "begin_request");
    if (request_id == kInvalidId) {
      reject(telemetry_.lifecycle_rejections, "invalid request identity");
    }
    clear_request_runtime();
    request_id_ = request_id;
    ++request_epoch_;
    lifecycle_ = Lifecycle::request_active;
    ++telemetry_.requests_begun;
  }

  void declare_lifetime(Id object_id, std::uint32_t first_consumer,
                        std::uint32_t last_consumer) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "declare_lifetime");
    const std::size_t object_index = find_object(object_id);
    const auto& object = manifest_.objects[object_index];
    auto& runtime = object_runtime_[object_index];
    if (runtime.scheduled || first_consumer > last_consumer ||
        last_consumer >= object.max_consumers) {
      reject(telemetry_.lifetime_rejections, "invalid or duplicate object lifetime");
    }
    runtime.scheduled = true;
    runtime.first_consumer = first_consumer;
    runtime.last_consumer = last_consumer;
    scheduled_object_indices_.push_back(object_index);
    ++telemetry_.scheduled_objects;
    telemetry_.scheduled_source_bytes += object.source.bytes;
  }

  WindowTicket start_read(Id object_id, Id requested_window_id = kInvalidId) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "start_read");
    const std::size_t object_index = find_object(object_id);
    const auto& object = manifest_.objects[object_index];
    auto& object_runtime = object_runtime_[object_index];
    if (!object_runtime.scheduled || object_runtime.read_started) {
      reject(telemetry_.lifetime_rejections,
             "object is not eligible for a host-window read");
    }
    const Id window_id = requested_window_id == kInvalidId
                             ? object.host_window_id
                             : requested_window_id;
    if (window_id == kInvalidId || !window_allowed(object, window_id)) {
      reject(telemetry_.wrong_object_rejections,
             "object read from an undeclared host window");
    }
    const std::size_t window_index = find_window(window_id);
    const auto& window_manifest = manifest_.host_windows[window_index];
    auto& window = window_runtime_[window_index];
    if (window.state != HostWindowState::free) {
      reject(telemetry_.premature_reuse_rejections,
             "host window was not explicitly recycled");
    }
    if (window_manifest.bytes < object.source.bytes) {
      reject(telemetry_.capacity_rejections,
             "object does not fit host window");
    }
    if (window.generation == std::numeric_limits<Id>::max()) {
      reject(telemetry_.capacity_rejections,
             "host-window generation exhausted");
    }
    ++window.generation;
    window.object_id = object_id;
    window.state = HostWindowState::filling;
    window.source_completion_event = 0;
    window.h2d_completion_event = 0;
    window.copy_consumer = kInvalidConsumer;
    window.live_consumers = 0;
    window.h2d_complete = false;
    object_runtime.read_started = true;
    object_runtime.window_id = window_id;
    object_runtime.window_generation = window.generation;
    ++telemetry_.source_reads_issued;
    return WindowTicket{request_epoch_, object_id, window_id, window.generation};
  }

  void complete_read(const WindowTicket& ticket, Id completion_event,
                     std::uint64_t completed_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "complete_read");
    const std::size_t object_index = validate_window_ticket(ticket, "source read");
    auto& window = window_runtime_[find_window(ticket.window_id)];
    const auto& object = manifest_.objects[object_index];
    if (completion_event == 0 || completed_bytes != object.source.bytes) {
      reject(telemetry_.missing_completion_rejections,
             "invalid source-read completion");
    }
    if (window.state != HostWindowState::filling) {
      if (window.source_completion_event == completion_event &&
          (window.state == HostWindowState::ready ||
           window.state == HostWindowState::copying ||
           window.state == HostWindowState::recyclable)) {
        return;
      }
      reject(telemetry_.duplicate_completion_rejections,
             "conflicting source-read completion");
    }
    window.state = HostWindowState::ready;
    window.source_completion_event = completion_event;
    ++telemetry_.source_reads_completed;
    telemetry_.completed_application_read_bytes += completed_bytes;
    telemetry_.storage_bytes += completed_bytes;
  }

  CopyTicket begin_copy(const WindowTicket& ticket, Id slot_id,
                        std::uint32_t consumer) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "begin_copy");
    const std::size_t object_index = validate_window_ticket(ticket, "H2D copy");
    auto& object_runtime = object_runtime_[object_index];
    const auto& object = manifest_.objects[object_index];
    auto& window = window_runtime_[find_window(ticket.window_id)];
    if (window.state != HostWindowState::ready ||
        window.source_completion_event == 0 || window.live_consumers != 0 ||
        consumer < object_runtime.first_consumer ||
        consumer > object_runtime.last_consumer ||
        object_runtime.completed[consumer] ||
        object_runtime.live_tickets[consumer] != kInvalidId) {
      reject(telemetry_.premature_reuse_rejections,
             "host window is not ready for the declared consumer");
    }
    if (!slot_allowed(object, slot_id)) {
      reject(telemetry_.wrong_object_rejections,
             "object copied to an undeclared GPU slot");
    }
    const Id slot_generation = bind_locked(slot_id, ticket.object_id);
    window.state = HostWindowState::copying;
    window.copy_consumer = consumer;
    window.h2d_completion_event = 0;
    window.h2d_complete = false;
    if (object_runtime.copies_issued != 0) {
      ++telemetry_.migration_reused_objects;
      telemetry_.migration_reused_bytes += object.source.bytes;
    }
    ++object_runtime.copies_issued;
    telemetry_.h2d_issued_bytes += object.source.bytes;
    return CopyTicket{request_epoch_, ticket.object_id, ticket.window_id,
                      ticket.generation, slot_id, slot_generation, consumer};
  }

  void complete_copy(const CopyTicket& ticket, Id completion_event,
                     std::uint64_t completed_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "complete_copy");
    const std::size_t object_index = validate_copy_ticket(ticket);
    auto& window = window_runtime_[find_window(ticket.window_id)];
    auto& slot = slot_runtime_[find_slot(ticket.slot_id)];
    const auto& object = manifest_.objects[object_index];
    if (window.state != HostWindowState::copying || window.h2d_complete ||
        window.copy_consumer != ticket.consumer || completion_event == 0 ||
        completed_bytes != object.source.bytes) {
      reject(telemetry_.missing_completion_rejections,
             "invalid H2D completion");
    }
    slot.ready = true;
    slot.readiness_event = completion_event;
    window.h2d_complete = true;
    window.h2d_completion_event = completion_event;
    ++telemetry_.readiness_events;
    telemetry_.h2d_completed_bytes += completed_bytes;
    telemetry_.h2d_bytes += completed_bytes;
  }

  void mark_unused(const WindowTicket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "mark_unused");
    const std::size_t object_index = validate_window_ticket(ticket, "unused object");
    auto& object_runtime = object_runtime_[object_index];
    auto& window = window_runtime_[find_window(ticket.window_id)];
    if (object_runtime.consumed || object_runtime.unused ||
        window.live_consumers != 0 ||
        (window.state != HostWindowState::ready &&
         !(window.state == HostWindowState::copying && window.h2d_complete))) {
      reject(telemetry_.premature_reuse_rejections,
             "object cannot be classified unused");
    }
    object_runtime.unused = true;
    window.state = HostWindowState::recyclable;
    ++telemetry_.unused_objects;
    telemetry_.speculative_unused_bytes +=
        manifest_.objects[object_index].source.bytes;
  }

  void seal_lifetime(Id object_id, std::uint32_t last_consumer) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "seal_lifetime");
    const std::size_t object_index = find_object(object_id);
    auto& object = object_runtime_[object_index];
    if (!object.scheduled || !object.consumed || object.unused ||
        last_consumer < object.first_consumer ||
        last_consumer > object.last_consumer) {
      reject(telemetry_.lifetime_rejections,
             "invalid sealed object lifetime");
    }
    for (std::uint32_t consumer = object.first_consumer;
         consumer <= last_consumer; ++consumer) {
      if (!object.completed[consumer] ||
          object.live_tickets[consumer] != kInvalidId) {
        reject(telemetry_.missing_completion_rejections,
               "sealed lifetime has an incomplete consumer");
      }
    }
    for (std::uint32_t consumer = last_consumer + 1;
         consumer <= object.last_consumer; ++consumer) {
      if (object.completed[consumer] ||
          object.live_tickets[consumer] != kInvalidId) {
        reject(telemetry_.lifetime_rejections,
               "sealed lifetime discards observed consumers");
      }
    }
    object.last_consumer = last_consumer;
    if (object.window_id != kInvalidId) {
      auto& window = window_runtime_[find_window(object.window_id)];
      if (window.object_id != object_id || window.live_consumers != 0 ||
          window.state != HostWindowState::ready) {
        reject(telemetry_.premature_reuse_rejections,
               "sealed lifetime host window is not ready");
      }
      window.state = HostWindowState::recyclable;
    }
  }

  void recycle_window(const WindowTicket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_operational("recycle_window");
    (void)validate_window_ticket(ticket, "window recycle");
    auto& window = window_runtime_[find_window(ticket.window_id)];
    if (window.state != HostWindowState::recyclable ||
        window.live_consumers != 0) {
      reject(telemetry_.premature_reuse_rejections,
             "host window has live or incomplete work");
    }
    clear_window(window);
    ++telemetry_.window_recycles;
  }

  void retire_window(const WindowTicket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "retire_window");
    const std::size_t object_index =
        validate_window_ticket(ticket, "window retirement");
    auto& object = object_runtime_[object_index];
    auto& window = window_runtime_[find_window(ticket.window_id)];
    if (window.live_consumers != 0 ||
        window.state == HostWindowState::filling ||
        window.state == HostWindowState::copying) {
      reject(telemetry_.premature_reuse_rejections,
             "host window has live or incomplete work");
    }
    if (!object.consumed) {
      if (object.unused || window.state != HostWindowState::ready) {
        reject(telemetry_.lifetime_rejections,
               "unconsumed host window cannot be retired");
      }
      object.unused = true;
      ++telemetry_.unused_objects;
      telemetry_.speculative_unused_bytes +=
          manifest_.objects[object_index].source.bytes;
    } else {
      std::uint32_t last_completed = kInvalidConsumer;
      bool gap = false;
      for (std::uint32_t consumer = object.first_consumer;
           consumer <= object.last_consumer; ++consumer) {
        if (object.live_tickets[consumer] != kInvalidId) {
          reject(telemetry_.missing_completion_rejections,
                 "retired lifetime has a live consumer");
        }
        if (object.completed[consumer]) {
          if (gap) {
            reject(telemetry_.lifetime_rejections,
                   "retired lifetime has a consumer gap");
          }
          last_completed = consumer;
        } else {
          gap = true;
        }
      }
      if (last_completed == kInvalidConsumer ||
          (window.state != HostWindowState::ready &&
           window.state != HostWindowState::recyclable)) {
        reject(telemetry_.missing_completion_rejections,
               "retired lifetime has no completed consumer");
      }
      object.last_consumer = last_completed;
    }
    window.state = HostWindowState::recyclable;
    clear_window(window);
    ++telemetry_.window_recycles;
  }

  void record_exposed_wait(std::uint64_t source_wait_ns,
                           std::uint64_t h2d_wait_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_operational("record_exposed_wait");
    telemetry_.exposed_source_wait_ns += source_wait_ns;
    telemetry_.exposed_h2d_wait_ns += h2d_wait_ns;
  }

  Id bind(Id slot_id, Id object_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_operational("bind");
    return bind_locked(slot_id, object_id);
  }

  void mark_ready(Id slot_id, Id generation, Id readiness_event) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_operational("mark_ready");
    auto& slot = slot_runtime_[find_slot(slot_id)];
    if (slot.object_id == kInvalidId) {
      reject(telemetry_.wrong_object_rejections, "cannot ready an unbound slot");
    }
    if (slot.generation != generation) {
      reject(telemetry_.stale_generation_rejections, "stale readiness generation");
    }
    if (readiness_event == 0 || slot.ready) {
      reject(telemetry_.missing_ready_rejections, "invalid or duplicate readiness event");
    }
    slot.ready = true;
    slot.readiness_event = readiness_event;
    ++telemetry_.readiness_events;
  }

  Ticket acquire(Id object_id, std::uint32_t consumer) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "acquire");
    const std::size_t object_index = find_object(object_id);
    auto& object = object_runtime_[object_index];
    if (!object.scheduled || consumer < object.first_consumer ||
        consumer > object.last_consumer) {
      reject(telemetry_.lifetime_rejections, "consumer is outside object lifetime");
    }
    if (object.completed[consumer] || object.live_tickets[consumer] != kInvalidId) {
      reject(telemetry_.premature_reuse_rejections, "consumer already acquired or completed");
    }
    const std::size_t slot_index = slot_for_object(object_id);
    auto& slot = slot_runtime_[slot_index];
    if (!slot.ready || slot.readiness_event == 0) {
      reject(telemetry_.missing_ready_rejections, "object is not ready");
    }
    Id window_id = kInvalidId;
    Id window_generation = 0;
    const auto& object_manifest = manifest_.objects[object_index];
    if (object.window_id != kInvalidId && object.window_generation != 0) {
      auto& window = window_runtime_[find_window(object.window_id)];
      if (window.state != HostWindowState::copying || !window.h2d_complete ||
          window.object_id != object_id ||
          window.generation != object.window_generation ||
          window.copy_consumer != consumer) {
        reject(telemetry_.missing_ready_rejections,
               "host window is not copy-complete for this consumer");
      }
      window_id = object.window_id;
      window_generation = window.generation;
      ++window.live_consumers;
    }
    const Id ticket_id = ++next_ticket_id_;
    object.live_tickets[consumer] = ticket_id;
    if (!object.consumed) {
      object.consumed = true;
      ++telemetry_.consumed_source_objects;
      telemetry_.consumer_used_source_bytes += object_manifest.source.bytes;
    }
    ++slot.live_consumers;
    ++live_consumers_;
    ++telemetry_.consumer_acquires;
    return Ticket{ticket_id, request_epoch_, object_id,
                  manifest_.gpu_slots[slot_index].slot_id,
                  slot.generation, consumer, window_id, window_generation};
  }

  void complete(const Ticket& ticket, Id completion_event) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "complete");
    if (completion_event == 0) {
      reject(telemetry_.missing_completion_rejections, "completion event is missing");
    }
    if (ticket.request_epoch != request_epoch_) {
      reject(telemetry_.stale_generation_rejections, "ticket request epoch is stale");
    }
    const std::size_t object_index = find_object(ticket.object_id);
    const std::size_t slot_index = find_slot(ticket.slot_id);
    auto& object = object_runtime_[object_index];
    auto& slot = slot_runtime_[slot_index];
    if (ticket.consumer >= object.live_tickets.size() ||
        object.live_tickets[ticket.consumer] == kInvalidId) {
      reject(telemetry_.duplicate_completion_rejections, "consumer is not live");
    }
    if (object.live_tickets[ticket.consumer] != ticket.ticket_id ||
        slot.object_id != ticket.object_id) {
      reject(telemetry_.wrong_object_rejections, "completion ticket identity mismatch");
    }
    if (slot.generation != ticket.generation) {
      reject(telemetry_.stale_generation_rejections, "completion generation is stale");
    }
    HostWindowRuntime* window = nullptr;
    if (ticket.window_id != kInvalidId) {
      window = &window_runtime_[find_window(ticket.window_id)];
      if (window->object_id != ticket.object_id ||
          window->generation != ticket.window_generation ||
          window->state != HostWindowState::copying ||
          window->copy_consumer != ticket.consumer ||
          window->live_consumers == 0) {
        reject(telemetry_.stale_generation_rejections,
               "completion host-window generation is stale");
      }
    }
    object.live_tickets[ticket.consumer] = kInvalidId;
    object.completed[ticket.consumer] = true;
    --slot.live_consumers;
    --live_consumers_;
    slot.completion_event = completion_event;
    if (window != nullptr) {
      --window->live_consumers;
      if (window->live_consumers == 0) {
        window->state = lifetime_complete(object)
                            ? HostWindowState::recyclable
                            : HostWindowState::ready;
        window->copy_consumer = kInvalidConsumer;
        window->h2d_complete = false;
      }
    }
    ++telemetry_.consumer_completions;
  }

  StateTicket stage_state(Id branch_id, Id parent_generation,
                          Id candidate_generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "stage_state");
    if (state_runtime_.empty() || branch_id == kInvalidId ||
        candidate_generation == committed_state_generation_ ||
        !state_parent_exists(parent_generation)) {
      reject(telemetry_.state_rejections, "invalid state lineage");
    }
    for (const auto& state : state_runtime_) {
      if (state.active && (state.branch_id == branch_id ||
                           state.candidate_generation == candidate_generation)) {
        reject(telemetry_.state_rejections, "duplicate branch or state generation");
      }
    }
    for (auto& state : state_runtime_) {
      if (!state.active) {
        state.active = true;
        state.transaction_id = ++next_state_transaction_id_;
        state.branch_id = branch_id;
        state.parent_generation = parent_generation;
        state.candidate_generation = candidate_generation;
        ++telemetry_.state_stages;
        return StateTicket{state.transaction_id, request_epoch_, branch_id,
                           parent_generation, candidate_generation};
      }
    }
    reject(telemetry_.capacity_rejections, "state transaction capacity exhausted");
  }

  void commit_state(const StateTicket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "commit_state");
    StateRuntime& selected = find_state(ticket);
    if (!state_lineage_exists(selected.parent_generation, 0)) {
      reject(telemetry_.state_rejections, "selected state lineage is incomplete");
    }
    const Id selected_transaction = selected.transaction_id;
    const Id selected_generation = selected.candidate_generation;
    for (auto& state : state_runtime_) {
      if (state.active && state.transaction_id != selected_transaction) {
        ++telemetry_.state_rollbacks;
      }
      state = {};
    }
    committed_state_generation_ = selected_generation;
    ++telemetry_.state_commits;
  }

  void rollback_state(const StateTicket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "rollback_state");
    StateRuntime& state = find_state(ticket);
    state = {};
    ++telemetry_.state_rollbacks;
  }

  void rollback_all_state() {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "rollback_all_state");
    for (auto& state : state_runtime_) {
      if (state.active) {
        state = {};
        ++telemetry_.state_rollbacks;
      }
    }
  }

  void finish_request() {
    std::lock_guard<std::mutex> lock(mutex_);
    require_lifecycle(Lifecycle::request_active, "finish_request");
    if (live_consumers_ != 0) {
      reject(telemetry_.missing_completion_rejections, "request has live consumers");
    }
    for (const std::size_t object_index : scheduled_object_indices_) {
      const auto& object = object_runtime_[object_index];
      if (object.unused) {
        continue;
      }
      for (std::uint32_t consumer = object.first_consumer;
           consumer <= object.last_consumer; ++consumer) {
        if (!object.completed[consumer]) {
          reject(telemetry_.missing_completion_rejections,
                 "declared lifetime was not completely consumed");
        }
      }
    }
    if (has_staged_state()) {
      reject(telemetry_.state_rejections, "request has uncommitted state");
    }
    for (const auto& window : window_runtime_) {
      if (window.state != HostWindowState::free &&
          window.state != HostWindowState::recyclable) {
        reject(telemetry_.missing_completion_rejections,
               "request has incomplete host-window work");
      }
    }
    clear_request_runtime();
    request_id_ = kInvalidId;
    lifecycle_ = Lifecycle::ready;
    ++telemetry_.requests_finished;
  }

  void reset(Id committed_state_generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lifecycle_ != Lifecycle::ready && lifecycle_ != Lifecycle::request_active) {
      reject(telemetry_.lifecycle_rejections, "reset requires an operational engine");
    }
    if (live_consumers_ != 0) {
      reject(telemetry_.premature_reuse_rejections, "reset with live consumers");
    }
    for (const auto& window : window_runtime_) {
      if (window.state == HostWindowState::filling ||
          window.state == HostWindowState::copying) {
        reject(telemetry_.premature_reuse_rejections,
               "reset with active host-window work");
      }
    }
    for (auto& state : state_runtime_) {
      if (state.active) {
        state = {};
        ++telemetry_.state_rollbacks;
      }
    }
    clear_request_runtime();
    for (std::size_t index = 0; index < slot_runtime_.size(); ++index) {
      auto& slot = slot_runtime_[index];
      if (slot.object_id != kInvalidId) {
        const auto object = object_index_.find(slot.object_id);
        if (object != object_index_.end() &&
            (manifest_.objects[object->second].host_window_id != kInvalidId ||
             !manifest_.objects[object->second]
                  .allowed_host_window_ids.empty())) {
          object_to_slot_[object->second] = kInvalidIndex;
          clear_slot(slot);
          ++telemetry_.slot_releases;
        }
      }
    }
    for (auto& window : window_runtime_) {
      if (window.state != HostWindowState::free) {
        clear_window(window);
        ++telemetry_.window_recycles;
      }
    }
    committed_state_generation_ = committed_state_generation;
    request_id_ = kInvalidId;
    ++request_epoch_;
    lifecycle_ = Lifecycle::ready;
    ++telemetry_.resets;
  }

  void release_slot(Id slot_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_operational("release_slot");
    auto& slot = slot_runtime_[find_slot(slot_id)];
    if (slot.object_id == kInvalidId) {
      reject(telemetry_.wrong_object_rejections, "slot is not bound");
    }
    if (slot.live_consumers != 0) {
      reject(telemetry_.premature_reuse_rejections, "slot has live consumers");
    }
    const auto object_it = object_index_.find(slot.object_id);
    if (object_it != object_index_.end()) {
      const auto& object = object_runtime_[object_it->second];
      const bool physical_copy_complete =
          object.window_id != kInvalidId && object.window_generation != 0;
      if (physical_copy_complete) {
        const auto& window =
            window_runtime_[find_window(object.window_id)];
        if (window.state == HostWindowState::filling ||
            window.state == HostWindowState::copying) {
          reject(telemetry_.premature_reuse_rejections,
                 "slot released before its physical consumer completed");
        }
      }
      if (object.scheduled && !physical_copy_complete) {
        for (std::uint32_t consumer = object.first_consumer;
             consumer <= object.last_consumer; ++consumer) {
          if (!object.completed[consumer]) {
            reject(telemetry_.premature_reuse_rejections,
                   "slot released before last consumer");
          }
        }
      }
      object_to_slot_[object_it->second] = kInvalidIndex;
    }
    clear_slot(slot);
    ++telemetry_.slot_releases;
  }

  void record_transfer(std::uint64_t storage_bytes, std::uint64_t h2d_bytes,
                       std::uint64_t d2h_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_operational("record_transfer");
    telemetry_.storage_bytes += storage_bytes;
    telemetry_.h2d_bytes += h2d_bytes;
    telemetry_.d2h_bytes += d2h_bytes;
  }

  void record_cost(const CostTerms& terms) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_operational("record_cost");
    if (terms.output_tokens <= 0.0 || terms.commitment <= 0.0 ||
        terms.prompt_ms < 0.0 || terms.proposer_ms < 0.0 ||
        terms.target_compute_ms < 0.0 || terms.storage_ms < 0.0 ||
        terms.h2d_ms < 0.0) {
      reject(telemetry_.capacity_rejections, "invalid request cost terms");
    }
    last_cost_ = terms;
    ++telemetry_.cost_records;
  }

  void shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lifecycle_ != Lifecycle::ready) {
      reject(telemetry_.lifecycle_rejections, "shutdown requires a ready engine");
    }
    if (live_consumers_ != 0 || has_staged_state()) {
      reject(telemetry_.premature_reuse_rejections, "shutdown with live work");
    }
    for (const auto& slot : slot_runtime_) {
      if (slot.object_id != kInvalidId) {
        reject(telemetry_.premature_reuse_rejections, "shutdown with a bound slot");
      }
    }
    for (const auto& window : window_runtime_) {
      if (window.state != HostWindowState::free) {
        reject(telemetry_.premature_reuse_rejections,
               "shutdown with a live host window");
      }
    }
    lifecycle_ = Lifecycle::shutdown;
    ++telemetry_.shutdowns;
  }

  Snapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot result;
    result.lifecycle = lifecycle_;
    result.manifest_identity = manifest_.manifest_identity;
    result.request_id = request_id_;
    result.request_epoch = request_epoch_;
    result.committed_state_generation = committed_state_generation_;
    result.live_consumers = live_consumers_;
    for (const auto& slot : slot_runtime_) {
      result.bound_slots += slot.object_id != kInvalidId ? 1 : 0;
      result.ready_slots += slot.ready ? 1 : 0;
    }
    for (const auto& state : state_runtime_) {
      result.staged_states += state.active ? 1 : 0;
    }
    for (const auto& window : window_runtime_) {
      switch (window.state) {
        case HostWindowState::free:
          ++result.free_windows;
          break;
        case HostWindowState::filling:
          ++result.filling_windows;
          break;
        case HostWindowState::ready:
          ++result.ready_windows;
          break;
        case HostWindowState::copying:
          ++result.copying_windows;
          break;
        case HostWindowState::recyclable:
          ++result.recyclable_windows;
          break;
      }
    }
    result.telemetry = telemetry_;
    result.last_cost = last_cost_;
    return result;
  }

  const Manifest& manifest() const { return manifest_; }

 private:
  struct ObjectRuntime {
    bool scheduled = false;
    bool read_started = false;
    bool consumed = false;
    bool unused = false;
    std::uint32_t first_consumer = 0;
    std::uint32_t last_consumer = 0;
    std::uint64_t copies_issued = 0;
    Id window_id = kInvalidId;
    Id window_generation = 0;
    std::vector<Id> live_tickets;
    std::vector<bool> completed;
  };

  struct SlotRuntime {
    Id object_id = kInvalidId;
    Id generation = 0;
    Id readiness_event = 0;
    Id completion_event = 0;
    std::uint64_t live_consumers = 0;
    bool ready = false;
  };

  struct HostWindowRuntime {
    HostWindowState state = HostWindowState::free;
    Id object_id = kInvalidId;
    Id generation = 0;
    Id source_completion_event = 0;
    Id h2d_completion_event = 0;
    std::uint32_t copy_consumer = kInvalidConsumer;
    std::uint64_t live_consumers = 0;
    bool h2d_complete = false;
  };

  struct StateRuntime {
    Id transaction_id = kInvalidId;
    Id branch_id = kInvalidId;
    Id parent_generation = 0;
    Id candidate_generation = 0;
    bool active = false;
  };

  [[noreturn]] void reject(std::uint64_t& counter, const char* message) {
    ++counter;
    throw std::logic_error(message);
  }

  void require_lifecycle(Lifecycle expected, const char* operation) {
    if (lifecycle_ != expected) {
      const std::string message = std::string(operation) + " is invalid in this lifecycle";
      reject(telemetry_.lifecycle_rejections, message.c_str());
    }
  }

  void require_operational(const char* operation) {
    if (lifecycle_ != Lifecycle::ready && lifecycle_ != Lifecycle::request_active) {
      const std::string message = std::string(operation) + " requires an operational engine";
      reject(telemetry_.lifecycle_rejections, message.c_str());
    }
  }

  Id bind_locked(Id slot_id, Id object_id) {
    const std::size_t slot_index = find_slot(slot_id);
    const std::size_t object_index = find_object(object_id);
    auto& slot = slot_runtime_[slot_index];
    const auto& slot_manifest = manifest_.gpu_slots[slot_index];
    const auto& object = manifest_.objects[object_index];
    if (slot.object_id != kInvalidId || slot.live_consumers != 0) {
      reject(telemetry_.premature_reuse_rejections,
             "slot is still bound or acquired");
    }
    if (slot_manifest.bytes < object.source.bytes) {
      reject(telemetry_.capacity_rejections, "object does not fit GPU slot");
    }
    if (!slot_allowed(object, slot_id)) {
      reject(telemetry_.wrong_object_rejections,
             "object bound to an undeclared GPU slot");
    }
    if (object_to_slot_[object_index] != kInvalidIndex) {
      reject(telemetry_.wrong_object_rejections, "object is already bound");
    }
    if (slot.generation == std::numeric_limits<Id>::max()) {
      reject(telemetry_.capacity_rejections, "slot generation exhausted");
    }
    ++slot.generation;
    slot.object_id = object_id;
    object_to_slot_[object_index] = slot_index;
    slot.ready = false;
    slot.readiness_event = 0;
    slot.completion_event = 0;
    ++telemetry_.bindings;
    return slot.generation;
  }

  void clear_slot(SlotRuntime& slot) {
    slot.object_id = kInvalidId;
    slot.ready = false;
    slot.readiness_event = 0;
    slot.completion_event = 0;
    slot.live_consumers = 0;
  }

  void clear_window(HostWindowRuntime& window) {
    window.state = HostWindowState::free;
    window.object_id = kInvalidId;
    window.source_completion_event = 0;
    window.h2d_completion_event = 0;
    window.copy_consumer = kInvalidConsumer;
    window.live_consumers = 0;
    window.h2d_complete = false;
  }

  bool lifetime_complete(const ObjectRuntime& object) const {
    if (!object.scheduled || object.unused) {
      return object.unused;
    }
    for (std::uint32_t consumer = object.first_consumer;
         consumer <= object.last_consumer; ++consumer) {
      if (!object.completed[consumer]) {
        return false;
      }
    }
    return true;
  }

  static bool extent_inside(std::uint64_t inner_address, std::uint64_t inner_bytes,
                            std::uint64_t outer_address, std::uint64_t outer_bytes) {
    if (inner_bytes == 0 || outer_bytes == 0 || inner_address < outer_address) {
      return false;
    }
    const std::uint64_t inner_offset = inner_address - outer_address;
    return inner_offset <= outer_bytes && inner_bytes <= outer_bytes - inner_offset;
  }

  static bool slot_allowed(const ObjectManifest& object, Id slot_id) {
    if (object.gpu_slot_id != kInvalidId) {
      return object.gpu_slot_id == slot_id;
    }
    if (!object.allowed_gpu_slot_ids.empty()) {
      return std::find(object.allowed_gpu_slot_ids.begin(),
                       object.allowed_gpu_slot_ids.end(),
                       slot_id) != object.allowed_gpu_slot_ids.end();
    }
    return true;
  }

  static bool window_allowed(const ObjectManifest& object, Id window_id) {
    if (object.host_window_id != kInvalidId) {
      return object.host_window_id == window_id;
    }
    return std::find(object.allowed_host_window_ids.begin(),
                     object.allowed_host_window_ids.end(),
                     window_id) != object.allowed_host_window_ids.end();
  }

  void validate_manifest() {
    if (manifest_.manifest_identity == kInvalidId ||
        manifest_.arena.arena_id == kInvalidId ||
        manifest_.arena.stable_address == 0 || manifest_.arena.bytes == 0 ||
        manifest_.objects.empty() || manifest_.gpu_slots.empty()) {
      throw std::invalid_argument("incomplete Plan-v0 manifest");
    }
    std::unordered_map<Id, bool> objects;
    for (const auto& object : manifest_.objects) {
      if (object.object_id == kInvalidId || object.tensor_id == kInvalidId ||
          object.graph_role_id == kInvalidId || object.source.source_id == kInvalidId ||
          object.source.extent_identity == kInvalidId || object.source.bytes == 0 ||
          object.max_consumers == 0 || !objects.emplace(object.object_id, true).second ||
          !extent_inside(object.stable_address, object.source.bytes,
                         manifest_.arena.stable_address, manifest_.arena.bytes)) {
        throw std::invalid_argument("invalid Plan-v0 object manifest");
      }
    }
    std::unordered_map<Id, bool> slots;
    for (const auto& slot : manifest_.gpu_slots) {
      if (slot.slot_id == kInvalidId || slot.bytes == 0 ||
          !slots.emplace(slot.slot_id, true).second) {
        throw std::invalid_argument("invalid Plan-v0 GPU slot manifest");
      }
    }
    std::unordered_map<Id, bool> windows;
    for (const auto& window : manifest_.host_windows) {
      if (window.window_id == kInvalidId || window.stable_address == 0 ||
          window.bytes == 0 || !windows.emplace(window.window_id, true).second) {
        throw std::invalid_argument("invalid Plan-v0 host window manifest");
      }
    }
    for (const auto& object : manifest_.objects) {
      if (object.host_window_id != kInvalidId &&
          !object.allowed_host_window_ids.empty()) {
        throw std::invalid_argument(
            "Plan-v0 object has conflicting host-window authorities");
      }
      if (object.gpu_slot_id != kInvalidId &&
          !object.allowed_gpu_slot_ids.empty()) {
        throw std::invalid_argument(
            "Plan-v0 object has conflicting GPU-slot authorities");
      }
      if (object.host_window_id != kInvalidId) {
        const auto window = std::find_if(
            manifest_.host_windows.begin(), manifest_.host_windows.end(),
            [&](const HostWindowManifest& value) {
              return value.window_id == object.host_window_id;
            });
        if (window == manifest_.host_windows.end() ||
            window->bytes < object.source.bytes) {
          throw std::invalid_argument("invalid Plan-v0 object host-window binding");
        }
      }
      std::unordered_map<Id, bool> allowed_windows;
      for (const Id window_id : object.allowed_host_window_ids) {
        const auto window = std::find_if(
            manifest_.host_windows.begin(), manifest_.host_windows.end(),
            [&](const HostWindowManifest& value) {
              return value.window_id == window_id;
            });
        if (window_id == kInvalidId ||
            !allowed_windows.emplace(window_id, true).second ||
            window == manifest_.host_windows.end() ||
            window->bytes < object.source.bytes) {
          throw std::invalid_argument(
              "invalid Plan-v0 object host-window pool binding");
        }
      }
      if (object.gpu_slot_id != kInvalidId) {
        const auto slot = std::find_if(
            manifest_.gpu_slots.begin(), manifest_.gpu_slots.end(),
            [&](const SlotManifest& value) {
              return value.slot_id == object.gpu_slot_id;
            });
        if (slot == manifest_.gpu_slots.end() ||
            slot->bytes < object.source.bytes) {
          throw std::invalid_argument("invalid Plan-v0 object GPU-slot binding");
        }
      }
      std::unordered_map<Id, bool> allowed_slots;
      for (const Id slot_id : object.allowed_gpu_slot_ids) {
        const auto slot = std::find_if(
            manifest_.gpu_slots.begin(), manifest_.gpu_slots.end(),
            [&](const SlotManifest& value) { return value.slot_id == slot_id; });
        if (slot_id == kInvalidId ||
            !allowed_slots.emplace(slot_id, true).second ||
            slot == manifest_.gpu_slots.end() ||
            slot->bytes < object.source.bytes) {
          throw std::invalid_argument(
              "invalid Plan-v0 object GPU-slot pool binding");
        }
      }
    }
  }

  std::size_t find_object(Id object_id) {
    const auto iterator = object_index_.find(object_id);
    if (iterator == object_index_.end()) {
      reject(telemetry_.wrong_object_rejections, "unknown object identity");
    }
    return iterator->second;
  }

  std::size_t find_slot(Id slot_id) {
    const auto iterator = slot_index_.find(slot_id);
    if (iterator == slot_index_.end()) {
      reject(telemetry_.wrong_object_rejections, "unknown slot identity");
    }
    return iterator->second;
  }

  std::size_t find_window(Id window_id) {
    const auto iterator = window_index_.find(window_id);
    if (iterator == window_index_.end()) {
      reject(telemetry_.wrong_object_rejections,
             "unknown host-window identity");
    }
    return iterator->second;
  }

  std::size_t validate_window_ticket(const WindowTicket& ticket,
                                     const char* operation) {
    if (ticket.request_epoch != request_epoch_) {
      reject(telemetry_.stale_generation_rejections,
             "host-window ticket request epoch is stale");
    }
    const std::size_t object_index = find_object(ticket.object_id);
    const auto& object = manifest_.objects[object_index];
    auto& runtime = object_runtime_[object_index];
    const auto& window = window_runtime_[find_window(ticket.window_id)];
    if (ticket.generation == 0 || window.generation != ticket.generation) {
      const std::string message = std::string(operation) +
                                  " host-window generation is stale";
      reject(telemetry_.stale_generation_rejections, message.c_str());
    }
    if (!window_allowed(object, ticket.window_id) ||
        window.object_id != ticket.object_id) {
      reject(telemetry_.wrong_object_rejections,
             "host-window ticket object identity mismatch");
    }
    if (runtime.window_generation != ticket.generation) {
      reject(telemetry_.stale_generation_rejections,
             "object host-window generation is stale");
    }
    return object_index;
  }

  std::size_t validate_copy_ticket(const CopyTicket& ticket) {
    const std::size_t object_index = validate_window_ticket(
        WindowTicket{ticket.request_epoch, ticket.object_id, ticket.window_id,
                     ticket.window_generation},
        "H2D completion");
    const auto& slot = slot_runtime_[find_slot(ticket.slot_id)];
    if (slot.object_id != ticket.object_id) {
      reject(telemetry_.wrong_object_rejections,
             "H2D ticket object identity mismatch");
    }
    if (slot.generation != ticket.slot_generation) {
      reject(telemetry_.stale_generation_rejections,
             "H2D slot generation is stale");
    }
    return object_index;
  }

  std::size_t slot_for_object(Id object_id) {
    const std::size_t object_index = find_object(object_id);
    const std::size_t slot_index = object_to_slot_[object_index];
    if (slot_index != kInvalidIndex) {
      return slot_index;
    }
    reject(telemetry_.missing_ready_rejections, "object is not bound");
  }

  bool state_parent_exists(Id generation) const {
    if (generation == committed_state_generation_) {
      return true;
    }
    for (const auto& state : state_runtime_) {
      if (state.active && state.candidate_generation == generation) {
        return true;
      }
    }
    return false;
  }

  bool state_lineage_exists(Id generation, std::size_t depth) const {
    if (generation == committed_state_generation_) {
      return true;
    }
    if (depth >= state_runtime_.size()) {
      return false;
    }
    for (const auto& state : state_runtime_) {
      if (state.active && state.candidate_generation == generation) {
        return state_lineage_exists(state.parent_generation, depth + 1);
      }
    }
    return false;
  }

  StateRuntime& find_state(const StateTicket& ticket) {
    if (ticket.request_epoch != request_epoch_) {
      reject(telemetry_.state_rejections, "stale state request epoch");
    }
    for (auto& state : state_runtime_) {
      if (state.active && state.transaction_id == ticket.transaction_id &&
          state.branch_id == ticket.branch_id &&
          state.parent_generation == ticket.parent_generation &&
          state.candidate_generation == ticket.candidate_generation) {
        return state;
      }
    }
    reject(telemetry_.state_rejections, "unknown or stale state transaction");
  }

  bool has_staged_state() const {
    return std::any_of(state_runtime_.begin(), state_runtime_.end(),
                       [](const StateRuntime& state) { return state.active; });
  }

  void clear_request_runtime() {
    for (const std::size_t object_index : scheduled_object_indices_) {
      ++telemetry_.request_runtime_object_visits;
      auto& object = object_runtime_[object_index];
      object.scheduled = false;
      object.read_started = false;
      object.consumed = false;
      object.unused = false;
      object.first_consumer = 0;
      object.last_consumer = 0;
      object.copies_issued = 0;
      object.window_id = kInvalidId;
      object.window_generation = 0;
      std::fill(object.live_tickets.begin(), object.live_tickets.end(), kInvalidId);
      std::fill(object.completed.begin(), object.completed.end(), false);
    }
    scheduled_object_indices_.clear();
  }

  static constexpr std::size_t kInvalidIndex =
      std::numeric_limits<std::size_t>::max();
  Manifest manifest_;
  mutable std::mutex mutex_;
  Lifecycle lifecycle_ = Lifecycle::constructed;
  Id request_id_ = kInvalidId;
  Id request_epoch_ = 0;
  Id committed_state_generation_ = 0;
  Id next_ticket_id_ = 0;
  Id next_state_transaction_id_ = 0;
  std::uint64_t live_consumers_ = 0;
  std::unordered_map<Id, std::size_t> object_index_;
  std::unordered_map<Id, std::size_t> slot_index_;
  std::unordered_map<Id, std::size_t> window_index_;
  std::vector<ObjectRuntime> object_runtime_;
  std::vector<SlotRuntime> slot_runtime_;
  std::vector<HostWindowRuntime> window_runtime_;
  std::vector<std::size_t> object_to_slot_;
  std::vector<std::size_t> scheduled_object_indices_;
  std::vector<StateRuntime> state_runtime_;
  Telemetry telemetry_;
  CostTerms last_cost_;
};

}  // namespace msi::plan_v0
