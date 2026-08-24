#include <torch/extension.h>
#include <pybind11/stl.h>

#include "plan_service_bundle.hpp"
#include "bounded_source_service.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef MSI_PLAN_SOURCE_SHA256
#error "MSI_PLAN_SOURCE_SHA256 must identify the compiled plan_v0.hpp"
#endif

#ifndef MSI_BUNDLE_SOURCE_SHA256
#error "MSI_BUNDLE_SOURCE_SHA256 must identify the compiled plan_service_bundle.hpp"
#endif

#define MSI_STRINGIFY_INNER(value) #value
#define MSI_STRINGIFY(value) MSI_STRINGIFY_INNER(value)

namespace py = pybind11;
using msi::plan_v0::Engine;
using msi::plan_v0::Id;
using msi::plan_v0::CopyTicket;
using msi::plan_v0::Ticket;
using msi::plan_v0::WindowTicket;
using msi::plan_service::BundleManifest;
using msi::plan_service::ComponentManifest;
using msi::plan_service::DestinationView;
using msi::plan_service::Extent;
using msi::plan_service::FileManifest;
using ServiceManifest = msi::plan_service::Manifest;
using BoundedSourceService = msi::bounded_source::Service;
using BoundedSourceTicket = msi::bounded_source::Ticket;

namespace {

constexpr char kPlanSourceSha256[] = MSI_STRINGIFY(MSI_PLAN_SOURCE_SHA256);
static_assert(sizeof(kPlanSourceSha256) == 65);
constexpr char kBundleSourceSha256[] = MSI_STRINGIFY(MSI_BUNDLE_SOURCE_SHA256);
static_assert(sizeof(kBundleSourceSha256) == 65);

std::uint64_t as_u64(const py::dict& row, const char* key) {
  return py::cast<std::uint64_t>(row[py::str(key)]);
}

std::string as_string(const py::handle& value) {
  return py::cast<std::string>(py::str(value));
}

class BoundedSourceBridge {
 public:
  BoundedSourceBridge(const std::string& path, const py::list& windows,
                      std::uint64_t max_in_flight, bool direct_io,
                      bool discard_buffered_cache = false) {
    std::vector<msi::bounded_source::Window> values;
    for (const py::handle item : windows) {
      const py::dict row = py::cast<py::dict>(item);
      values.push_back({as_u64(row, "window_id"), as_u64(row, "address"),
                        as_u64(row, "bytes")});
    }
    service_ = std::make_unique<BoundedSourceService>(
        path, std::move(values), max_in_flight, direct_io, 4096,
        discard_buffered_cache);
  }

  std::uint64_t submit(std::uint64_t window_id, const py::list& extents) {
    std::vector<msi::bounded_source::Extent> values;
    for (const py::handle item : extents) {
      const py::dict row = py::cast<py::dict>(item);
      values.push_back({as_u64(row, "source_offset"), as_u64(row, "bytes"),
                        as_u64(row, "destination_offset")});
    }
    BoundedSourceTicket ticket = service_->submit(window_id, std::move(values));
    tickets_.emplace(ticket.ticket_id, ticket);
    return ticket.ticket_id;
  }

  void await(std::uint64_t ticket_id) {
    const BoundedSourceTicket ticket = find(ticket_id);
    py::gil_scoped_release release;
    service_->await(ticket);
  }

  void cancel(std::uint64_t ticket_id) {
    const BoundedSourceTicket ticket = find(ticket_id);
    py::gil_scoped_release release;
    service_->cancel(ticket);
  }

  void begin_h2d(std::uint64_t ticket_id, std::uint64_t bytes) {
    service_->begin_h2d(find(ticket_id), bytes);
  }

  void complete_h2d(std::uint64_t ticket_id, std::uint64_t bytes,
                    std::uint64_t event_id) {
    service_->complete_h2d(find(ticket_id), bytes, event_id);
  }

  void mark_retirable(std::uint64_t ticket_id) {
    service_->mark_retirable(find(ticket_id));
  }

  void retire(std::uint64_t ticket_id) {
    service_->retire(find(ticket_id));
    tickets_.erase(ticket_id);
  }

  py::dict snapshot() const {
    const auto value = service_->snapshot();
    py::dict result;
    result["submissions"] = value.submissions;
    result["completions"] = value.completions;
    result["cancellations"] = value.cancellations;
    result["failures"] = value.failures;
    result["logical_bytes"] = value.logical_bytes;
    result["physical_read_bytes"] = value.physical_read_bytes;
    result["block_read_bytes"] = value.block_read_bytes;
    result["padding_bytes"] = value.padding_bytes;
    result["read_wall_ns"] = value.read_wall_ns;
    result["exposed_wait_ns"] = value.exposed_wait_ns;
    result["h2d_issued_bytes"] = value.h2d_issued_bytes;
    result["h2d_completed_bytes"] = value.h2d_completed_bytes;
    result["queue_rejections"] = value.queue_rejections;
    result["lifecycle_rejections"] = value.lifecycle_rejections;
    result["generation_reuses"] = value.generation_reuses;
    result["current_in_flight"] = value.current_in_flight;
    result["peak_in_flight"] = value.peak_in_flight;
    result["active_tickets"] = value.active_tickets;
    result["free_windows"] = value.free_windows;
    result["filling_windows"] = value.filling_windows;
    result["ready_windows"] = value.ready_windows;
    result["copying_windows"] = value.copying_windows;
    result["retirable_windows"] = value.retirable_windows;
    result["source_inventory_bytes"] = service_->source_bytes();
    result["max_in_flight"] = service_->max_in_flight();
    result["direct_io"] = service_->direct_io();
    return result;
  }

  void shutdown() { service_->shutdown(); }

 private:
  const BoundedSourceTicket& find(std::uint64_t ticket_id) const {
    const auto ticket = tickets_.find(ticket_id);
    if (ticket == tickets_.end()) {
      throw std::logic_error("unknown bounded source bridge ticket");
    }
    return ticket->second;
  }

  std::unique_ptr<BoundedSourceService> service_;
  std::unordered_map<std::uint64_t, BoundedSourceTicket> tickets_;
};

class PlanServiceBridge {
 public:
  PlanServiceBridge(const py::dict& manifest, std::uint64_t arena_address,
                    std::uint64_t host_address,
                    const std::vector<std::uint64_t>& gpu_slot_addresses = {},
                    const std::vector<std::uint64_t>& host_window_addresses = {}) {
    if (arena_address == 0 || host_address == 0) {
      throw std::invalid_argument("Plan bridge requires stable arena addresses");
    }
    ServiceManifest service;
    service.manifest_identity = intern(manifest["artifact"]);
    const py::dict resources = py::cast<py::dict>(manifest["resources"]);
    service.arena = msi::plan_v0::ArenaManifest{
        intern("arena"), arena_address,
        as_u64(resources, "managed_gpu_capacity_bytes")};
    service.managed_host_bytes =
        as_u64(resources, "managed_host_capacity_bytes");

    for (const py::handle item : py::cast<py::list>(manifest["files"])) {
      const py::dict row = py::cast<py::dict>(item);
      service.files.push_back(
          FileManifest{intern(row["file_id"]), as_u64(row, "bytes")});
    }

    const py::list host_windows = py::cast<py::list>(resources["host_windows"]);
    if (!host_window_addresses.empty() &&
        host_window_addresses.size() != host_windows.size()) {
      throw std::invalid_argument("host-window address count differs from manifest");
    }
    std::uint64_t host_offset = 0;
    std::size_t host_index = 0;
    for (const py::handle item : host_windows) {
      const py::dict row = py::cast<py::dict>(item);
      const std::uint64_t bytes = as_u64(row, "bytes");
      const std::string public_id = as_string(row["window_id"]);
      const Id internal_id = intern(row["window_id"]);
      const std::uint64_t address = host_window_addresses.empty()
                                        ? host_address + host_offset
                                        : host_window_addresses[host_index];
      if (address == 0) {
        throw std::invalid_argument("host window requires a stable address");
      }
      service.host_windows.push_back(msi::plan_v0::HostWindowManifest{
          internal_id, address, bytes});
      window_ids_.emplace(public_id, internal_id);
      window_views_.emplace(public_id, View{address, bytes});
      host_offset += bytes;
      ++host_index;
    }
    const py::list gpu_slots = py::cast<py::list>(resources["gpu_slots"]);
    if (!gpu_slot_addresses.empty() &&
        gpu_slot_addresses.size() != gpu_slots.size()) {
      throw std::invalid_argument("GPU slot address count differs from manifest");
    }
    std::uint64_t gpu_offset = 0;
    std::size_t gpu_index = 0;
    for (const py::handle item : gpu_slots) {
      const py::dict row = py::cast<py::dict>(item);
      const std::string public_id = as_string(row["slot_id"]);
      const Id internal_id = intern(row["slot_id"]);
      const std::uint64_t bytes = as_u64(row, "bytes");
      const std::uint64_t address = gpu_slot_addresses.empty()
                                        ? arena_address + gpu_offset
                                        : gpu_slot_addresses[gpu_index];
      if (address == 0) {
        throw std::invalid_argument("GPU slot requires a stable address");
      }
      slot_ids_.emplace(public_id, internal_id);
      service.gpu_slots.push_back(
          msi::plan_v0::SlotManifest{internal_id, bytes, address});
      slot_views_.emplace(public_id, View{address, bytes});
      gpu_offset += bytes;
      ++gpu_index;
    }

    const py::dict identities = py::cast<py::dict>(manifest["identities"]);
    const py::dict plan_identity =
        py::cast<py::dict>(identities["plan_engine"]);
    if (as_string(plan_identity["sha256"]) != kPlanSourceSha256) {
      throw std::invalid_argument("compiled Plan source identity mismatch");
    }
    for (const py::handle item : py::cast<py::list>(manifest["objects"])) {
      const py::dict row = py::cast<py::dict>(item);
      const py::dict requirements = py::cast<py::dict>(row["requirements"]);
      const py::dict bindings = py::cast<py::dict>(row["identity_bindings"]);
      const std::string public_object_id = as_string(row["object_id"]);
      BundleManifest object;
      object.object_id = intern(row["object_id"]);
      object.graph_role_id = row.contains("graph_role_id")
                                 ? intern(row["graph_role_id"])
                                 : intern(row["bundle_id"]);
      const std::uint64_t arena_offset =
          requirements.contains("arena_offset")
              ? as_u64(requirements, "arena_offset")
              : 0;
      object.stable_address = arena_address + arena_offset;
      object.max_consumers =
          py::cast<std::uint32_t>(row["maximum_consumer_count"]);
      object.canonical_artifact_id =
          intern(py::cast<py::dict>(bindings["canonical_artifact"])["identity"]);
      object.transformation_id =
          intern(py::cast<py::dict>(bindings["transformation"])["identity"]);
      object.bundle_identity = intern(row["bundle_id"]);
      object.execution_layout_id =
          intern(py::cast<py::dict>(bindings["execution_layout"])["identity"]);
      object.backend_id =
          intern(py::cast<py::dict>(bindings["backend"])["identity"]);
      if (requirements.contains("slot_id")) {
        object.slot_id = intern(requirements["slot_id"]);
      } else if (requirements.contains("allowed_slot_ids")) {
        for (const py::handle slot_id :
             py::cast<py::list>(requirements["allowed_slot_ids"])) {
          object.allowed_slot_ids.push_back(intern(slot_id));
        }
      }
      if (requirements.contains("window_id")) {
        object.host_window_id = intern(requirements["window_id"]);
      } else if (requirements.contains("allowed_window_ids")) {
        for (const py::handle window_id :
             py::cast<py::list>(requirements["allowed_window_ids"])) {
          object.allowed_host_window_ids.push_back(intern(window_id));
        }
      }
      object.materialized_bytes = as_u64(requirements, "bundle_bytes");
      object.slot_requirement_bytes = as_u64(requirements, "slot_bytes");
      object.window_requirement_bytes = as_u64(requirements, "window_bytes");
      object.required_alignment = as_u64(requirements, "alignment");

      for (const py::handle tensor_id :
           py::cast<py::list>(row["canonical_tensor_ids"])) {
        object.canonical_tensor_ids.push_back(intern(tensor_id));
      }
      for (const py::handle role_id :
           py::cast<py::list>(row["required_component_roles"])) {
        object.required_role_ids.push_back(intern(role_id));
      }

      for (const py::handle component_item :
           py::cast<py::list>(row["components"])) {
        const py::dict component_row = py::cast<py::dict>(component_item);
        const py::dict source = py::cast<py::dict>(component_row["source"]);
        const py::dict materialized =
            py::cast<py::dict>(component_row["materialized"]);
        const py::dict destination =
            py::cast<py::dict>(component_row["destination_view"]);
        const Id component_id = intern(component_row["component_id"]);
        ComponentManifest component;
        component.component_id = component_id;
        component.role_id = intern(component_row["role_id"]);
        component.canonical_tensor_id =
            intern(component_row["canonical_tensor_id"]);
        component.source = Extent{
            intern(source["file_id"]), as_u64(source, "offset"),
            as_u64(source, "bytes"), as_u64(source, "alignment"),
            intern(public_object_id + ":" +
                   as_string(component_row["component_id"]) + ":source")};
        component.materialized = Extent{
            intern(materialized["file_id"]), as_u64(materialized, "offset"),
            as_u64(materialized, "bytes"),
            as_u64(materialized, "alignment"),
            intern(public_object_id + ":" +
                   as_string(component_row["component_id"]) + ":materialized")};
        component.destination = DestinationView{
            intern(destination["view_id"]), as_u64(destination, "offset"),
            as_u64(destination, "bytes"), as_u64(destination, "alignment")};
        component.quantization_block_bytes =
            py::cast<std::uint64_t>(component_row["quantization_block_bytes"]);
        component.required = py::cast<bool>(component_row["required"]);
        for (const py::handle alias :
             py::cast<py::list>(component_row["allow_alias_with"])) {
          component.allow_alias_with.push_back(intern(alias));
        }
        object.components.push_back(component);
        views_.emplace(
            view_key(public_object_id, as_string(component_row["component_id"])),
            View{object.stable_address + component.destination.offset,
                 component.destination.bytes});
      }
      object_ids_.emplace(public_object_id, object.object_id);
      service.bundles.push_back(std::move(object));
    }
    service.max_state_transactions = 8;
    engine_ = std::make_unique<Engine>(
        msi::plan_service::lower_to_plan_v0(service));
  }

  void initialize(Id state_generation = 0) { engine_->initialize(state_generation); }

  void begin_request(const std::string& request_id) {
    engine_->begin_request(intern(request_id));
  }

  void declare_lifetime(const std::string& object_id, std::uint32_t first,
                        std::uint32_t last) {
    engine_->declare_lifetime(find_object(object_id), first, last);
  }

  Id bind(const std::string& slot_id, const std::string& object_id) {
    return engine_->bind(find_slot(slot_id), find_object(object_id));
  }

  Id start_read(const std::string& object_id,
                const std::string& window_id = "") {
    const Id ticket_id = ++next_bridge_ticket_id_;
    window_tickets_.emplace(
        ticket_id,
        engine_->start_read(find_object(object_id),
                            window_id.empty() ? msi::plan_v0::kInvalidId
                                              : find_window(window_id)));
    return ticket_id;
  }

  void complete_read(Id ticket_id, Id event_id, std::uint64_t bytes) {
    engine_->complete_read(find_window_ticket(ticket_id), event_id, bytes);
  }

  Id begin_copy(Id window_ticket_id, const std::string& slot_id,
                std::uint32_t consumer) {
    const Id copy_ticket_id = ++next_bridge_ticket_id_;
    copy_tickets_.emplace(
        copy_ticket_id,
        engine_->begin_copy(find_window_ticket(window_ticket_id),
                            find_slot(slot_id), consumer));
    return copy_ticket_id;
  }

  void complete_copy(Id ticket_id, Id event_id, std::uint64_t bytes) {
    const auto iterator = copy_tickets_.find(ticket_id);
    if (iterator == copy_tickets_.end()) {
      throw std::logic_error("unknown bridge copy ticket");
    }
    engine_->complete_copy(iterator->second, event_id, bytes);
    copy_tickets_.erase(iterator);
  }

  void mark_unused(Id ticket_id) {
    engine_->mark_unused(find_window_ticket(ticket_id));
  }

  void seal_lifetime(const std::string& object_id,
                     std::uint32_t last_consumer) {
    engine_->seal_lifetime(find_object(object_id), last_consumer);
  }

  void recycle_window(Id ticket_id) {
    const auto iterator = window_tickets_.find(ticket_id);
    if (iterator == window_tickets_.end()) {
      throw std::logic_error("unknown bridge window ticket");
    }
    engine_->recycle_window(iterator->second);
    window_tickets_.erase(iterator);
  }

  void retire_window(Id ticket_id) {
    const auto iterator = window_tickets_.find(ticket_id);
    if (iterator == window_tickets_.end()) {
      throw std::logic_error("unknown bridge window ticket");
    }
    engine_->retire_window(iterator->second);
    window_tickets_.erase(iterator);
  }

  void mark_ready(const std::string& slot_id, Id generation, Id event_id) {
    engine_->mark_ready(find_slot(slot_id), generation, event_id);
  }

  Id acquire(const std::string& object_id, std::uint32_t consumer) {
    Ticket ticket = engine_->acquire(find_object(object_id), consumer);
    const Id ticket_id = ticket.ticket_id;
    tickets_.emplace(ticket_id, std::move(ticket));
    return ticket_id;
  }

  void complete(Id ticket_id, Id event_id) {
    const auto iterator = tickets_.find(ticket_id);
    if (iterator == tickets_.end()) {
      throw std::logic_error("unknown bridge ticket");
    }
    engine_->complete(iterator->second, event_id);
    tickets_.erase(iterator);
  }

  void finish_request() { engine_->finish_request(); }

  void release_slot(const std::string& slot_id) {
    engine_->release_slot(find_slot(slot_id));
  }

  void record_transfer(std::uint64_t storage_bytes, std::uint64_t h2d_bytes,
                       std::uint64_t d2h_bytes = 0) {
    engine_->record_transfer(storage_bytes, h2d_bytes, d2h_bytes);
  }

  void record_exposed_wait(std::uint64_t source_wait_ns,
                           std::uint64_t h2d_wait_ns) {
    engine_->record_exposed_wait(source_wait_ns, h2d_wait_ns);
  }

  void reset(Id state_generation) {
    engine_->reset(state_generation);
    tickets_.clear();
    copy_tickets_.clear();
    window_tickets_.clear();
  }

  void shutdown() { engine_->shutdown(); }

  py::dict component_view(const std::string& object_id,
                          const std::string& component_id) const {
    const auto iterator = views_.find(view_key(object_id, component_id));
    if (iterator == views_.end()) {
      throw std::invalid_argument("unknown component view");
    }
    py::dict result;
    result["address"] = iterator->second.address;
    result["bytes"] = iterator->second.bytes;
    return result;
  }

  py::dict host_window_view(const std::string& window_id) const {
    const auto iterator = window_views_.find(window_id);
    if (iterator == window_views_.end()) {
      throw std::invalid_argument("unknown host-window view");
    }
    py::dict result;
    result["address"] = iterator->second.address;
    result["bytes"] = iterator->second.bytes;
    return result;
  }

  py::dict gpu_slot_view(const std::string& slot_id) const {
    const auto iterator = slot_views_.find(slot_id);
    if (iterator == slot_views_.end()) {
      throw std::invalid_argument("unknown GPU-slot view");
    }
    py::dict result;
    result["address"] = iterator->second.address;
    result["bytes"] = iterator->second.bytes;
    return result;
  }

  py::dict snapshot() const {
    const auto value = engine_->snapshot();
    py::dict telemetry;
    telemetry["requests_begun"] = value.telemetry.requests_begun;
    telemetry["requests_finished"] = value.telemetry.requests_finished;
    telemetry["bindings"] = value.telemetry.bindings;
    telemetry["slot_releases"] = value.telemetry.slot_releases;
    telemetry["readiness_events"] = value.telemetry.readiness_events;
    telemetry["consumer_acquires"] = value.telemetry.consumer_acquires;
    telemetry["consumer_completions"] = value.telemetry.consumer_completions;
    telemetry["resets"] = value.telemetry.resets;
    telemetry["shutdowns"] = value.telemetry.shutdowns;
    telemetry["storage_bytes"] = value.telemetry.storage_bytes;
    telemetry["h2d_bytes"] = value.telemetry.h2d_bytes;
    telemetry["d2h_bytes"] = value.telemetry.d2h_bytes;
    telemetry["scheduled_objects"] = value.telemetry.scheduled_objects;
    telemetry["scheduled_source_bytes"] =
        value.telemetry.scheduled_source_bytes;
    telemetry["source_reads_issued"] = value.telemetry.source_reads_issued;
    telemetry["source_reads_completed"] =
        value.telemetry.source_reads_completed;
    telemetry["completed_application_read_bytes"] =
        value.telemetry.completed_application_read_bytes;
    telemetry["h2d_issued_bytes"] = value.telemetry.h2d_issued_bytes;
    telemetry["h2d_completed_bytes"] = value.telemetry.h2d_completed_bytes;
    telemetry["consumed_source_objects"] =
        value.telemetry.consumed_source_objects;
    telemetry["consumer_used_source_bytes"] =
        value.telemetry.consumer_used_source_bytes;
    telemetry["migration_reused_objects"] =
        value.telemetry.migration_reused_objects;
    telemetry["migration_reused_bytes"] =
        value.telemetry.migration_reused_bytes;
    telemetry["unused_objects"] = value.telemetry.unused_objects;
    telemetry["speculative_unused_bytes"] =
        value.telemetry.speculative_unused_bytes;
    telemetry["window_recycles"] = value.telemetry.window_recycles;
    telemetry["exposed_source_wait_ns"] =
        value.telemetry.exposed_source_wait_ns;
    telemetry["exposed_h2d_wait_ns"] =
        value.telemetry.exposed_h2d_wait_ns;
    telemetry["request_runtime_object_visits"] =
        value.telemetry.request_runtime_object_visits;
    telemetry["missing_ready_rejections"] =
        value.telemetry.missing_ready_rejections;
    telemetry["stale_generation_rejections"] =
        value.telemetry.stale_generation_rejections;
    telemetry["wrong_object_rejections"] =
        value.telemetry.wrong_object_rejections;
    telemetry["premature_reuse_rejections"] =
        value.telemetry.premature_reuse_rejections;
    telemetry["missing_completion_rejections"] =
        value.telemetry.missing_completion_rejections;
    telemetry["duplicate_completion_rejections"] =
        value.telemetry.duplicate_completion_rejections;
    telemetry["lifetime_rejections"] = value.telemetry.lifetime_rejections;
    telemetry["lifecycle_rejections"] = value.telemetry.lifecycle_rejections;
    telemetry["capacity_rejections"] = value.telemetry.capacity_rejections;
    telemetry["state_rejections"] = value.telemetry.state_rejections;
    py::dict result;
    result["lifecycle"] = static_cast<int>(value.lifecycle);
    result["request_epoch"] = value.request_epoch;
    result["bound_slots"] = value.bound_slots;
    result["ready_slots"] = value.ready_slots;
    result["live_consumers"] = value.live_consumers;
    result["staged_states"] = value.staged_states;
    result["free_windows"] = value.free_windows;
    result["filling_windows"] = value.filling_windows;
    result["ready_windows"] = value.ready_windows;
    result["copying_windows"] = value.copying_windows;
    result["recyclable_windows"] = value.recyclable_windows;
    result["bridge_window_tickets"] = window_tickets_.size();
    result["bridge_copy_tickets"] = copy_tickets_.size();
    result["telemetry"] = std::move(telemetry);
    result["plan_source_sha256"] = kPlanSourceSha256;
    result["bundle_source_sha256"] = kBundleSourceSha256;
    return result;
  }

 private:
  struct View {
    std::uint64_t address;
    std::uint64_t bytes;
  };

  static std::string view_key(const std::string& object_id,
                              const std::string& component_id) {
    return object_id + "\n" + component_id;
  }

  Id intern(const py::handle& value) { return intern(as_string(value)); }

  Id intern(const std::string& value) {
    const auto iterator = interned_.find(value);
    if (iterator != interned_.end()) {
      return iterator->second;
    }
    const Id identity = next_identity_++;
    interned_.emplace(value, identity);
    return identity;
  }

  Id find_object(const std::string& value) const {
    const auto iterator = object_ids_.find(value);
    if (iterator == object_ids_.end()) {
      throw std::invalid_argument("unknown bridge object identity");
    }
    return iterator->second;
  }

  Id find_slot(const std::string& value) const {
    const auto iterator = slot_ids_.find(value);
    if (iterator == slot_ids_.end()) {
      throw std::invalid_argument("unknown bridge slot identity");
    }
    return iterator->second;
  }

  Id find_window(const std::string& value) const {
    const auto iterator = window_ids_.find(value);
    if (iterator == window_ids_.end()) {
      throw std::invalid_argument("unknown bridge host-window identity");
    }
    return iterator->second;
  }

  const WindowTicket& find_window_ticket(Id ticket_id) const {
    const auto iterator = window_tickets_.find(ticket_id);
    if (iterator == window_tickets_.end()) {
      throw std::logic_error("unknown bridge window ticket");
    }
    return iterator->second;
  }

  Id next_identity_ = 1;
  Id next_bridge_ticket_id_ = 0;
  std::unordered_map<std::string, Id> interned_;
  std::unordered_map<std::string, Id> object_ids_;
  std::unordered_map<std::string, Id> slot_ids_;
  std::unordered_map<std::string, Id> window_ids_;
  std::unordered_map<std::string, View> views_;
  std::unordered_map<std::string, View> window_views_;
  std::unordered_map<std::string, View> slot_views_;
  std::unordered_map<Id, Ticket> tickets_;
  std::unordered_map<Id, WindowTicket> window_tickets_;
  std::unordered_map<Id, CopyTicket> copy_tickets_;
  std::unique_ptr<Engine> engine_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<BoundedSourceBridge>(module, "BoundedSourceBridge")
      .def(py::init<const std::string&, const py::list&, std::uint64_t, bool, bool>(),
           py::arg("path"), py::arg("windows"), py::arg("max_in_flight"),
           py::arg("direct_io"), py::arg("discard_buffered_cache") = false)
      .def("submit", &BoundedSourceBridge::submit)
      .def("await_read", &BoundedSourceBridge::await)
      .def("cancel", &BoundedSourceBridge::cancel)
      .def("begin_h2d", &BoundedSourceBridge::begin_h2d)
      .def("complete_h2d", &BoundedSourceBridge::complete_h2d)
      .def("mark_retirable", &BoundedSourceBridge::mark_retirable)
      .def("retire", &BoundedSourceBridge::retire)
      .def("snapshot", &BoundedSourceBridge::snapshot)
      .def("shutdown", &BoundedSourceBridge::shutdown);
  py::class_<PlanServiceBridge>(module, "PlanServiceBridge")
      .def(py::init<const py::dict&, std::uint64_t, std::uint64_t,
                    const std::vector<std::uint64_t>&,
                    const std::vector<std::uint64_t>&>(),
           py::arg("manifest"), py::arg("arena_address"),
           py::arg("host_address"),
           py::arg("gpu_slot_addresses") = std::vector<std::uint64_t>{},
           py::arg("host_window_addresses") = std::vector<std::uint64_t>{})
      .def("initialize", &PlanServiceBridge::initialize, py::arg("state_generation") = 0)
      .def("begin_request", &PlanServiceBridge::begin_request)
      .def("declare_lifetime", &PlanServiceBridge::declare_lifetime)
      .def("start_read", &PlanServiceBridge::start_read,
           py::arg("object_id"), py::arg("window_id") = "")
      .def("complete_read", &PlanServiceBridge::complete_read)
      .def("begin_copy", &PlanServiceBridge::begin_copy)
      .def("complete_copy", &PlanServiceBridge::complete_copy)
      .def("mark_unused", &PlanServiceBridge::mark_unused)
      .def("seal_lifetime", &PlanServiceBridge::seal_lifetime)
      .def("recycle_window", &PlanServiceBridge::recycle_window)
      .def("retire_window", &PlanServiceBridge::retire_window)
      .def("bind", &PlanServiceBridge::bind)
      .def("mark_ready", &PlanServiceBridge::mark_ready)
      .def("acquire", &PlanServiceBridge::acquire)
      .def("complete", &PlanServiceBridge::complete)
      .def("finish_request", &PlanServiceBridge::finish_request)
      .def("release_slot", &PlanServiceBridge::release_slot)
      .def("record_transfer", &PlanServiceBridge::record_transfer,
           py::arg("storage_bytes"), py::arg("h2d_bytes"),
           py::arg("d2h_bytes") = 0)
      .def("record_exposed_wait", &PlanServiceBridge::record_exposed_wait)
      .def("reset", &PlanServiceBridge::reset)
      .def("shutdown", &PlanServiceBridge::shutdown)
      .def("component_view", &PlanServiceBridge::component_view)
      .def("host_window_view", &PlanServiceBridge::host_window_view)
      .def("gpu_slot_view", &PlanServiceBridge::gpu_slot_view)
      .def("snapshot", &PlanServiceBridge::snapshot);
  module.def("plan_source_sha256", [] { return std::string(kPlanSourceSha256); });
  module.def("bundle_source_sha256",
             [] { return std::string(kBundleSourceSha256); });
}
