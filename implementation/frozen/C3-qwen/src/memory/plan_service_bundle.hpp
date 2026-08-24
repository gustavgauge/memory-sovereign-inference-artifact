#pragma once

#include "plan_v0.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace msi::plan_service {

using Id = plan_v0::Id;
constexpr Id kInvalidId = plan_v0::kInvalidId;

struct FileManifest {
  Id file_id = kInvalidId;
  std::uint64_t bytes = 0;
};

struct Extent {
  Id file_id = kInvalidId;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 1;
  Id extent_identity = kInvalidId;
};

struct DestinationView {
  Id view_id = kInvalidId;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 1;
};

struct ComponentManifest {
  Id component_id = kInvalidId;
  Id role_id = kInvalidId;
  Id canonical_tensor_id = kInvalidId;
  Extent source;
  Extent materialized;
  DestinationView destination;
  std::uint64_t quantization_block_bytes = 1;
  bool required = true;
  std::vector<Id> allow_alias_with;
};

struct BundleManifest {
  Id object_id = kInvalidId;
  Id graph_role_id = kInvalidId;
  Id canonical_artifact_id = kInvalidId;
  Id transformation_id = kInvalidId;
  Id bundle_identity = kInvalidId;
  Id execution_layout_id = kInvalidId;
  Id backend_id = kInvalidId;
  Id slot_id = kInvalidId;
  Id host_window_id = kInvalidId;
  std::vector<Id> allowed_host_window_ids;
  std::uint64_t stable_address = 0;
  std::uint64_t materialized_bytes = 0;
  std::uint64_t slot_requirement_bytes = 0;
  std::uint64_t window_requirement_bytes = 0;
  std::uint64_t required_alignment = 1;
  std::uint32_t max_consumers = 1;
  std::vector<Id> canonical_tensor_ids;
  std::vector<Id> required_role_ids;
  std::vector<ComponentManifest> components;
  std::vector<Id> allowed_slot_ids;
};

struct Manifest {
  Id manifest_identity = kInvalidId;
  plan_v0::ArenaManifest arena;
  std::uint64_t managed_host_bytes = 0;
  std::vector<FileManifest> files;
  std::vector<BundleManifest> bundles;
  std::vector<plan_v0::SlotManifest> gpu_slots;
  std::vector<plan_v0::HostWindowManifest> host_windows;
  std::uint32_t max_state_transactions = 0;
};

inline bool extent_inside(std::uint64_t offset, std::uint64_t bytes,
                          std::uint64_t outer_bytes) {
  return offset <= outer_bytes && bytes <= outer_bytes - offset;
}

inline bool ranges_overlap(std::uint64_t left_offset, std::uint64_t left_bytes,
                           std::uint64_t right_offset,
                           std::uint64_t right_bytes) {
  return left_offset < right_offset + right_bytes &&
         right_offset < left_offset + left_bytes;
}

inline bool contains(const std::vector<Id>& values, Id value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

inline void validate_manifest(const Manifest& manifest) {
  if (manifest.manifest_identity == kInvalidId ||
      manifest.arena.arena_id == kInvalidId ||
      manifest.arena.stable_address == 0 || manifest.arena.bytes == 0 ||
      manifest.managed_host_bytes == 0 || manifest.files.empty() ||
      manifest.bundles.empty() || manifest.gpu_slots.empty() ||
      manifest.host_windows.empty()) {
    throw std::invalid_argument("incomplete service-bundle manifest");
  }

  std::unordered_map<Id, std::uint64_t> files;
  for (const auto& file : manifest.files) {
    if (file.file_id == kInvalidId || file.bytes == 0 ||
        !files.emplace(file.file_id, file.bytes).second) {
      throw std::invalid_argument("invalid service-bundle file manifest");
    }
  }

  std::unordered_map<Id, std::uint64_t> slots;
  std::uint64_t slot_sum = 0;
  for (const auto& slot : manifest.gpu_slots) {
    if (slot.slot_id == kInvalidId || slot.bytes == 0 ||
        !slots.emplace(slot.slot_id, slot.bytes).second) {
      throw std::invalid_argument("invalid service-bundle GPU slot");
    }
    slot_sum += slot.bytes;
  }
  if (slot_sum != manifest.arena.bytes) {
    throw std::invalid_argument("service-bundle GPU capacity sum mismatch");
  }

  std::unordered_map<Id, std::uint64_t> windows;
  std::uint64_t window_sum = 0;
  for (const auto& window : manifest.host_windows) {
    if (window.window_id == kInvalidId || window.stable_address == 0 ||
        window.bytes == 0 ||
        !windows.emplace(window.window_id, window.bytes).second) {
      throw std::invalid_argument("invalid service-bundle host window");
    }
    window_sum += window.bytes;
  }
  if (window_sum != manifest.managed_host_bytes) {
    throw std::invalid_argument("service-bundle host capacity sum mismatch");
  }

  struct CheckedExtent {
    Id component_id;
    Extent extent;
    std::vector<Id> aliases;
  };
  std::unordered_map<Id, std::vector<CheckedExtent>> source_extents;
  std::unordered_map<Id, std::vector<CheckedExtent>> materialized_extents;
  std::unordered_set<Id> object_ids;
  std::unordered_set<Id> component_ids;
  std::unordered_set<Id> view_ids;

  for (const auto& bundle : manifest.bundles) {
    if (bundle.object_id == kInvalidId ||
        !object_ids.emplace(bundle.object_id).second ||
        bundle.graph_role_id == kInvalidId ||
        bundle.canonical_artifact_id == kInvalidId ||
        bundle.transformation_id == kInvalidId ||
        bundle.bundle_identity == kInvalidId ||
        bundle.execution_layout_id == kInvalidId ||
        bundle.backend_id == kInvalidId ||
        (bundle.slot_id == kInvalidId) == bundle.allowed_slot_ids.empty() ||
        (bundle.host_window_id == kInvalidId) ==
            bundle.allowed_host_window_ids.empty() ||
        bundle.stable_address == 0 ||
        bundle.materialized_bytes == 0 || bundle.required_alignment == 0 ||
        bundle.stable_address % bundle.required_alignment != 0 ||
        bundle.max_consumers == 0 || bundle.canonical_tensor_ids.empty() ||
        bundle.required_role_ids.empty() || bundle.components.empty() ||
        bundle.stable_address < manifest.arena.stable_address ||
        !extent_inside(bundle.stable_address - manifest.arena.stable_address,
                       bundle.materialized_bytes, manifest.arena.bytes)) {
      throw std::invalid_argument("invalid atomic service bundle");
    }
    if (bundle.slot_requirement_bytes < bundle.materialized_bytes ||
        bundle.window_requirement_bytes < bundle.materialized_bytes) {
      throw std::invalid_argument("service-bundle capacity failure");
    }
    std::unordered_set<Id> allowed_windows;
    if (bundle.host_window_id != kInvalidId) {
      allowed_windows.emplace(bundle.host_window_id);
    } else {
      allowed_windows.insert(bundle.allowed_host_window_ids.begin(),
                             bundle.allowed_host_window_ids.end());
      if (allowed_windows.size() != bundle.allowed_host_window_ids.size()) {
        throw std::invalid_argument(
            "duplicate service-bundle host-window pool member");
      }
    }
    for (const Id window_id : allowed_windows) {
      const auto window = windows.find(window_id);
      if (window == windows.end() ||
          window->second < bundle.window_requirement_bytes) {
        throw std::invalid_argument(
            "service-bundle host-window pool capacity failure");
      }
    }
    std::unordered_set<Id> allowed_slots;
    if (bundle.slot_id != kInvalidId) {
      allowed_slots.emplace(bundle.slot_id);
    } else {
      allowed_slots.insert(bundle.allowed_slot_ids.begin(),
                           bundle.allowed_slot_ids.end());
      if (allowed_slots.size() != bundle.allowed_slot_ids.size()) {
        throw std::invalid_argument("duplicate service-bundle GPU slot pool member");
      }
    }
    for (const Id slot_id : allowed_slots) {
      const auto slot = slots.find(slot_id);
      if (slot == slots.end() || slot->second < bundle.slot_requirement_bytes) {
        throw std::invalid_argument("service-bundle GPU slot-pool capacity failure");
      }
    }

    std::unordered_set<Id> required_roles;
    std::vector<DestinationView> destinations;
    for (const auto& component : bundle.components) {
      if (component.component_id == kInvalidId ||
          !component_ids.emplace(component.component_id).second ||
          component.role_id == kInvalidId ||
          component.canonical_tensor_id == kInvalidId ||
          !contains(bundle.canonical_tensor_ids, component.canonical_tensor_id) ||
          component.quantization_block_bytes == 0 ||
          component.source.file_id == kInvalidId ||
          component.materialized.file_id == kInvalidId ||
          component.destination.view_id == kInvalidId ||
          !view_ids.emplace(component.destination.view_id).second ||
          component.source.bytes == 0 || component.materialized.bytes == 0 ||
          component.source.alignment == 0 ||
          component.materialized.alignment == 0 ||
          component.destination.alignment == 0 ||
          component.source.offset % component.source.alignment != 0 ||
          component.materialized.offset % component.materialized.alignment != 0 ||
          component.destination.offset % component.destination.alignment != 0 ||
          component.source.bytes % component.quantization_block_bytes != 0 ||
          component.materialized.bytes % component.quantization_block_bytes != 0 ||
          component.materialized.bytes != component.destination.bytes ||
          files.find(component.source.file_id) == files.end() ||
          files.find(component.materialized.file_id) == files.end() ||
          !extent_inside(component.source.offset, component.source.bytes,
                         files.at(component.source.file_id)) ||
          !extent_inside(component.materialized.offset,
                         component.materialized.bytes,
                         files.at(component.materialized.file_id)) ||
          !extent_inside(component.destination.offset,
                         component.destination.bytes,
                         bundle.materialized_bytes)) {
        throw std::invalid_argument("invalid service-bundle component");
      }
      for (const auto& destination : destinations) {
        if (ranges_overlap(component.destination.offset,
                           component.destination.bytes, destination.offset,
                           destination.bytes)) {
          throw std::invalid_argument("overlapping service-bundle views");
        }
      }
      destinations.push_back(component.destination);
      if (component.required) {
        required_roles.emplace(component.role_id);
      }
      source_extents[component.source.file_id].push_back(
          {component.component_id, component.source,
           component.allow_alias_with});
      materialized_extents[component.materialized.file_id].push_back(
          {component.component_id, component.materialized,
           component.allow_alias_with});
    }
    if (required_roles.size() != bundle.required_role_ids.size() ||
        std::any_of(bundle.required_role_ids.begin(),
                    bundle.required_role_ids.end(), [&](Id role) {
                      return required_roles.find(role) == required_roles.end();
                    })) {
      throw std::invalid_argument("incomplete atomic service bundle");
    }
  }

  const auto validate_overlaps = [](const auto& by_file) {
    for (const auto& [file_id, extents] : by_file) {
      (void)file_id;
      auto ordered = extents;
      std::sort(ordered.begin(), ordered.end(), [](const auto& left,
                                                   const auto& right) {
        return std::pair{left.extent.offset, left.extent.bytes} <
               std::pair{right.extent.offset, right.extent.bytes};
      });
      for (std::size_t right = 0; right < ordered.size(); ++right) {
        for (std::size_t left = right; left-- > 0;) {
          if (ordered[left].extent.offset + ordered[left].extent.bytes <=
              ordered[right].extent.offset) {
            break;
          }
          if (!contains(ordered[left].aliases,
                        ordered[right].component_id) ||
              !contains(ordered[right].aliases,
                        ordered[left].component_id)) {
            throw std::invalid_argument("undeclared service-bundle alias");
          }
        }
      }
    }
  };
  validate_overlaps(source_extents);
  validate_overlaps(materialized_extents);
}

inline plan_v0::Manifest lower_to_plan_v0(const Manifest& manifest) {
  validate_manifest(manifest);
  plan_v0::Manifest plan;
  plan.manifest_identity = manifest.manifest_identity;
  plan.arena = manifest.arena;
  plan.gpu_slots = manifest.gpu_slots;
  plan.host_windows = manifest.host_windows;
  plan.max_state_transactions = manifest.max_state_transactions;
  plan.objects.reserve(manifest.bundles.size());
  for (const auto& bundle : manifest.bundles) {
    plan.objects.push_back(plan_v0::ObjectManifest{
        bundle.object_id,
        bundle.canonical_tensor_ids.front(),
        bundle.graph_role_id,
        bundle.stable_address,
        plan_v0::SourceExtent{bundle.bundle_identity, 0,
                              bundle.materialized_bytes,
                              bundle.bundle_identity},
        bundle.max_consumers,
        bundle.host_window_id,
        bundle.allowed_host_window_ids,
        bundle.slot_id,
        bundle.allowed_slot_ids,
    });
  }
  return plan;
}

}  // namespace msi::plan_service
