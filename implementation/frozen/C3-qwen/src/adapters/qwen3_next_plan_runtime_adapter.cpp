#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-model.h"
#include "llama.h"

#include "plan_service_bundle.hpp"
#include "bounded_source_service.hpp"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
using msi::plan_service::BundleManifest;
using msi::plan_service::ComponentManifest;
using msi::plan_service::DestinationView;
using msi::plan_service::Extent;
using msi::plan_service::FileManifest;
using msi::plan_service::Manifest;
using msi::plan_v0::Engine;
using msi::plan_v0::Id;
using msi::plan_v0::Snapshot;
using PlanWindowTicket = msi::plan_v0::WindowTicket;
using SourceExtent = msi::bounded_source::Extent;
using SourceFaultMode = msi::bounded_source::FaultMode;
using SourceService = msi::bounded_source::Service;
using SourceTelemetry = msi::bounded_source::Telemetry;
using SourceTicket = msi::bounded_source::Ticket;
using SourceWindow = msi::bounded_source::Window;

constexpr std::uint64_t kOfficialModelBytes = 48410988384ULL;
constexpr int kLayers = 48;
constexpr int kExperts = 512;
constexpr int kExpertsUsed = 10;
constexpr std::uint64_t kSlotId = 300;
constexpr std::uint64_t kWindowId = 400;
constexpr int kMaximumSourceInFlight = 8;
constexpr std::uint64_t kSourceFileId = 10;
constexpr std::uint64_t kMaterializedFileId = 11;
constexpr std::uint64_t kGateRole = 3001;
constexpr std::uint64_t kDownRole = 3002;
constexpr std::uint64_t kUpRole = 3003;

struct Args {
    std::string arm;
    std::string fault = "none";
    std::string model;
    std::string backend_dir;
    std::string prompt;
    std::string prompt_file;
    std::string output;
    std::string logits;
    int n_predict = 32;
    int cache_capacity = 0;
    int n_gpu_layers = 99;
    int n_ubatch = 512;
    int prompt_token_limit = 0;
    bool preflight = false;
    bool drop_source_cache = false;
    bool no_mmap = false;
    bool direct_model_load = false;
    bool disable_thinking = false;
    bool bounded_source = false;
    bool source_direct = false;
    bool fixed_horizon = false;
    int source_in_flight = 1;
};

enum class SourceMode {
    Legacy,
    KernelPageCache,
    OracleBuffered,
    MinimalDirect,
    FullMsi,
};

bool is_campaign_arm(const Args & args) {
    return args.arm == "kernel_page_cache" || args.arm == "sync_oracle" ||
           args.arm == "minimal_direct" || args.arm == "full_msi";
}

bool is_fault_arm(const Args & args) {
    return args.arm == "fault_canary";
}

bool is_streamed_arm(const Args & args) {
    return args.arm == "plan_staged_candidate" ||
           args.arm == "plan_cached_candidate" || is_campaign_arm(args) ||
           is_fault_arm(args);
}

SourceMode source_mode(const Args & args) {
    if (args.arm == "kernel_page_cache") {
        return SourceMode::KernelPageCache;
    }
    if (args.arm == "sync_oracle") {
        return SourceMode::OracleBuffered;
    }
    if (args.arm == "minimal_direct") {
        return SourceMode::MinimalDirect;
    }
    if (args.arm == "full_msi") {
        return SourceMode::FullMsi;
    }
    if (is_fault_arm(args)) {
        return SourceMode::FullMsi;
    }
    return SourceMode::Legacy;
}

bool uses_full_msi(const Args & args) {
    return args.arm == "full_msi" ||
           args.arm == "plan_staged_candidate" ||
           args.arm == "plan_cached_candidate" || is_fault_arm(args);
}

const std::array<const char *, 10> kFaultModes{{
    "none",
    "short_read",
    "eio",
    "out_of_order",
    "stale_completion",
    "wrong_object",
    "held_consumer",
    "reset_outstanding",
    "shutdown_live_consumer",
    "partial_bundle",
}};

bool valid_fault_mode(const std::string & value) {
    return std::find_if(kFaultModes.begin(), kFaultModes.end(),
                        [&value](const char * mode) { return value == mode; }) !=
           kFaultModes.end();
}

SourceFaultMode source_fault_mode(const Args & args) {
    if (args.fault == "short_read") {
        return SourceFaultMode::ShortSuccess;
    }
    if (args.fault == "eio") {
        return SourceFaultMode::IoError;
    }
    if (args.fault == "partial_bundle") {
        return SourceFaultMode::PartialBundle;
    }
    return SourceFaultMode::None;
}

const char * source_mode_name(SourceMode mode) {
    switch (mode) {
        case SourceMode::KernelPageCache: return "kernel_page_cache";
        case SourceMode::OracleBuffered: return "sync_oracle_buffered_discard";
        case SourceMode::MinimalDirect: return "minimal_synchronous_odirect";
        case SourceMode::FullMsi: return "full_msi_async_odirect";
        case SourceMode::Legacy: return "legacy";
    }
    return "unknown";
}

struct TensorGeometry {
    std::uint64_t source_offset = 0;
    std::uint64_t tensor_bytes = 0;
    std::uint64_t expert_bytes = 0;
    ggml_type native_type = GGML_TYPE_COUNT;
    ggml_tensor * destination = nullptr;
};

struct LayerGeometry {
    TensorGeometry gate;
    TensorGeometry up;
    TensorGeometry down;
};

struct Timings {
    std::uint64_t source_ns = 0;
    std::uint64_t h2d_ns = 0;
    std::uint64_t scatter_ns = 0;
    std::uint64_t cache_fill_ns = 0;
};

struct PhaseSource {
    std::uint64_t logical_bytes = 0;
    std::uint64_t physical_read_bytes = 0;
    std::uint64_t read_wall_ns = 0;
    std::uint64_t exposed_wait_ns = 0;
    std::uint64_t h2d_completed_bytes = 0;
};

struct SimpleSourceTelemetry {
    std::uint64_t submissions = 0;
    std::uint64_t completions = 0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t physical_read_bytes = 0;
    std::uint64_t padding_bytes = 0;
    std::uint64_t read_wall_ns = 0;
    std::uint64_t h2d_issued_bytes = 0;
    std::uint64_t h2d_completed_bytes = 0;
    std::uint64_t fixed_direct_reads = 0;
};

Json phase_source_json(const PhaseSource & value) {
    return {
        {"logical_bytes", value.logical_bytes},
        {"physical_read_bytes", value.physical_read_bytes},
        {"read_wall_ns", value.read_wall_ns},
        {"exposed_wait_ns", value.exposed_wait_ns},
        {"h2d_completed_bytes", value.h2d_completed_bytes},
    };
}

struct ProcessIo {
    std::uint64_t rchar = 0;
    std::uint64_t syscr = 0;
    std::uint64_t read_bytes = 0;
};

void require(bool condition, const std::string & message);

ProcessIo process_io() {
    std::ifstream input("/proc/self/io");
    require(static_cast<bool>(input), "cannot open /proc/self/io");
    ProcessIo result;
    std::string key;
    std::uint64_t value = 0;
    while (input >> key >> value) {
        if (key == "rchar:") {
            result.rchar = value;
        } else if (key == "syscr:") {
            result.syscr = value;
        } else if (key == "read_bytes:") {
            result.read_bytes = value;
        }
    }
    return result;
}

Json process_io_delta(const ProcessIo & begin, const ProcessIo & end) {
    require(end.rchar >= begin.rchar && end.syscr >= begin.syscr &&
            end.read_bytes >= begin.read_bytes,
            "process I/O counters moved backwards");
    return {
        {"logical_read_characters", end.rchar - begin.rchar},
        {"read_syscalls", end.syscr - begin.syscr},
        {"physical_storage_read_bytes", end.read_bytes - begin.read_bytes},
    };
}

std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint64_t address_of(const void * pointer) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint64_t file_size(const std::string & path) {
    struct stat info {};
    if (stat(path.c_str(), &info) != 0) {
        throw std::runtime_error("stat failed for " + path + ": " + std::strerror(errno));
    }
    return static_cast<std::uint64_t>(info.st_size);
}

void read_exact(int fd, std::uint64_t offset, void * destination, std::size_t bytes) {
    std::size_t complete = 0;
    while (complete < bytes) {
        const ssize_t count = pread(fd, static_cast<char *>(destination) + complete,
                                    bytes - complete,
                                    static_cast<off_t>(offset + complete));
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("pread failed: " + std::string(std::strerror(errno)));
        }
        if (count == 0) {
            throw std::runtime_error("unexpected end of GGUF source");
        }
        complete += static_cast<std::size_t>(count);
    }
}

Args parse_args(int argc, char ** argv) {
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--preflight") {
            args.preflight = true;
            continue;
        }
        if (key == "--drop-source-cache") {
            args.drop_source_cache = true;
            continue;
        }
        if (key == "--no-mmap") {
            args.no_mmap = true;
            continue;
        }
        if (key == "--direct-model-load") {
            args.direct_model_load = true;
            continue;
        }
        if (key == "--disable-thinking") {
            args.disable_thinking = true;
            continue;
        }
        if (key == "--bounded-source") {
            args.bounded_source = true;
            continue;
        }
        if (key == "--source-direct") {
            args.source_direct = true;
            continue;
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + key);
        }
        const std::string value = argv[++index];
        if (key == "--arm") {
            args.arm = value;
        } else if (key == "--fault") {
            args.fault = value;
        } else if (key == "--model") {
            args.model = value;
        } else if (key == "--backend-dir") {
            args.backend_dir = value;
        } else if (key == "--prompt") {
            args.prompt = value;
        } else if (key == "--prompt-file") {
            args.prompt_file = value;
        } else if (key == "--output") {
            args.output = value;
        } else if (key == "--logits") {
            args.logits = value;
        } else if (key == "--n-predict") {
            args.n_predict = std::stoi(value);
        } else if (key == "--cache-capacity") {
            args.cache_capacity = std::stoi(value);
        } else if (key == "--n-gpu-layers") {
            args.n_gpu_layers = std::stoi(value);
        } else if (key == "--n-ubatch") {
            args.n_ubatch = std::stoi(value);
        } else if (key == "--prompt-token-limit") {
            args.prompt_token_limit = std::stoi(value);
        } else if (key == "--source-in-flight") {
            args.source_in_flight = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.arm != "resident_control" && args.arm != "resident_observed_control" &&
        args.arm != "resident_trace_control" && args.arm != "plan_staged_candidate" &&
        args.arm != "plan_cached_candidate" && args.arm != "kernel_page_cache" &&
        args.arm != "sync_oracle" && args.arm != "minimal_direct" &&
        args.arm != "full_msi" && args.arm != "fault_canary") {
        throw std::invalid_argument(
            "--arm must name resident_control, resident_observed_control, "
            "resident_trace_control, plan_staged_candidate, plan_cached_candidate, "
            "kernel_page_cache, sync_oracle, minimal_direct, full_msi, or fault_canary");
    }
    if (args.model.empty() || args.backend_dir.empty() ||
        (args.prompt.empty() == args.prompt_file.empty()) ||
        args.output.empty() || args.logits.empty() || args.n_predict < 8 ||
        args.n_predict > 128 || args.n_gpu_layers < 0 || args.n_ubatch < 1 ||
        args.n_ubatch > 8192 || args.prompt_token_limit < 0) {
        throw std::invalid_argument("incomplete or invalid Phase 3 arguments");
    }
    if (is_campaign_arm(args)) {
        if (args.cache_capacity != 64) {
            throw std::invalid_argument("campaign arms require --cache-capacity 64");
        }
        args.no_mmap = true;
        args.direct_model_load = false;
        args.disable_thinking = false;
        args.bounded_source = args.arm == "full_msi";
        args.source_direct = args.arm == "full_msi" ||
                             args.arm == "minimal_direct";
        args.source_in_flight = args.arm == "full_msi" ? 8 : 1;
        args.drop_source_cache = args.arm == "sync_oracle";
        args.fixed_horizon = true;
    } else if (is_fault_arm(args)) {
        if (!valid_fault_mode(args.fault) || args.cache_capacity != 0 ||
            args.prompt_token_limit != 512 || args.n_predict != 8 ||
            args.n_ubatch != 512) {
            throw std::invalid_argument(
                "fault_canary requires a declared fault, zero cache, 512+8, and n_ubatch 512");
        }
        args.no_mmap = true;
        args.direct_model_load = false;
        args.disable_thinking = false;
        args.bounded_source = true;
        args.source_direct = true;
        args.source_in_flight = 8;
        args.drop_source_cache = false;
        args.fixed_horizon = true;
    } else if ((args.arm == "plan_cached_candidate") != (args.cache_capacity > 0) ||
               args.cache_capacity < 0 || args.cache_capacity > kExperts) {
        throw std::invalid_argument(
            "plan_cached_candidate requires --cache-capacity in [1, 128]; other arms require 0");
    }
    if (args.direct_model_load &&
        (args.no_mmap || is_streamed_arm(args))) {
        throw std::invalid_argument(
            "--direct-model-load is exclusive with --no-mmap and is resident-only");
    }
    if (args.disable_thinking &&
        is_streamed_arm(args)) {
        throw std::invalid_argument("--disable-thinking is resident-only");
    }
    if (args.source_direct && !args.bounded_source &&
        args.arm != "minimal_direct") {
        throw std::invalid_argument("--source-direct requires --bounded-source");
    }
    if (args.source_in_flight < 1 || args.source_in_flight > kMaximumSourceInFlight ||
        (!args.bounded_source && args.source_in_flight != 1)) {
        throw std::invalid_argument(
            "bounded source in-flight count must be in [1, 8]");
    }
    if (!is_fault_arm(args) && args.fault != "none") {
        throw std::invalid_argument("--fault is exclusive to fault_canary");
    }
    return args;
}

std::string tensor_name(int layer, const char * suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

Id object_id(int layer, int expert) {
    return 100000 + static_cast<Id>(layer * kExperts + expert);
}

Json snapshot_json(const Snapshot & snapshot) {
    const auto & telemetry = snapshot.telemetry;
    return {
        {"lifecycle", static_cast<unsigned>(snapshot.lifecycle)},
        {"request_epoch", snapshot.request_epoch},
        {"ready_slots", snapshot.ready_slots},
        {"bound_slots", snapshot.bound_slots},
        {"live_consumers", snapshot.live_consumers},
        {"free_windows", snapshot.free_windows},
        {"filling_windows", snapshot.filling_windows},
        {"ready_windows", snapshot.ready_windows},
        {"copying_windows", snapshot.copying_windows},
        {"recyclable_windows", snapshot.recyclable_windows},
        {"telemetry", {
            {"requests_begun", telemetry.requests_begun},
            {"requests_finished", telemetry.requests_finished},
            {"scheduled_objects", telemetry.scheduled_objects},
            {"bindings", telemetry.bindings},
            {"readiness_events", telemetry.readiness_events},
            {"consumer_acquires", telemetry.consumer_acquires},
            {"consumer_completions", telemetry.consumer_completions},
            {"slot_releases", telemetry.slot_releases},
            {"source_reads_issued", telemetry.source_reads_issued},
            {"source_reads_completed", telemetry.source_reads_completed},
            {"completed_application_read_bytes", telemetry.completed_application_read_bytes},
            {"h2d_issued_bytes", telemetry.h2d_issued_bytes},
            {"h2d_completed_bytes", telemetry.h2d_completed_bytes},
            {"window_recycles", telemetry.window_recycles},
            {"resets", telemetry.resets},
            {"shutdowns", telemetry.shutdowns},
            {"lifecycle_rejections", telemetry.lifecycle_rejections},
            {"capacity_rejections", telemetry.capacity_rejections},
            {"stale_generation_rejections", telemetry.stale_generation_rejections},
            {"premature_reuse_rejections", telemetry.premature_reuse_rejections},
            {"missing_ready_rejections", telemetry.missing_ready_rejections},
            {"missing_completion_rejections", telemetry.missing_completion_rejections},
            {"wrong_object_rejections", telemetry.wrong_object_rejections},
            {"duplicate_completion_rejections", telemetry.duplicate_completion_rejections},
        }},
    };
}

class Qwen3NextRouteObserver {
  public:
    static bool callback(ggml_tensor * tensor, bool ask, void * user_data) {
        auto * self = static_cast<Qwen3NextRouteObserver *>(user_data);
        int layer = -1;
        if (sscanf(ggml_get_name(tensor), "ffn_moe_topk-%d", &layer) != 1 ||
            layer < 0 || layer >= kLayers) {
            return false;
        }
        if (ask) {
            return true;
        }
        self->observe(layer, tensor);
        return true;
    }

    void begin_step(std::uint64_t step) {
        require(layers_seen_.empty(), "route observer step already active");
        current_step_ = step;
        current_chunk_ = 0;
    }

    void finish_step() {
        require(layers_seen_.size() == kLayers,
                "route observer did not visit every Qwen3-Next MoE layer");
        layers_seen_.clear();
    }

    Json report() const {
        return {
            {"layers", kLayers},
            {"experts", kExperts},
            {"experts_used", kExpertsUsed},
            {"route_events", route_events_},
        };
    }

  private:
    void observe(int layer, ggml_tensor * selected_tensor) {
        if (layers_seen_.count(layer) != 0) {
            require(layers_seen_.size() == kLayers,
                    "route observer saw an incomplete repeated graph chunk");
            layers_seen_.clear();
            ++current_chunk_;
        }
        require(layers_seen_.insert(layer).second,
                "route observer could not bind the graph chunk layer");
        const std::size_t count = static_cast<std::size_t>(ggml_nelements(selected_tensor));
        require(count > 0 && count % kExpertsUsed == 0,
                "unexpected Qwen3-Next top-k tensor shape");
        std::vector<std::int32_t> selected(count);
        ggml_backend_tensor_get(selected_tensor, selected.data(), 0,
                                selected.size() * sizeof(selected.front()));
        std::vector<int> unique;
        std::array<bool, kExperts> seen{};
        for (const std::int32_t expert : selected) {
            require(expert >= 0 && expert < kExperts,
                    "router selected an invalid expert");
            if (!seen[expert]) {
                seen[expert] = true;
                unique.push_back(expert);
            }
        }
        route_events_.push_back({
            {"step", current_step_},
            {"chunk", current_chunk_},
            {"layer", layer},
            {"tokens", count / kExpertsUsed},
            {"selected_unique", unique},
        });
    }

    std::uint64_t current_step_ = 0;
    std::uint64_t current_chunk_ = 0;
    std::set<int> layers_seen_;
    Json route_events_ = Json::array();
};

class Qwen3NextPlanStreamer {
  public:
    Qwen3NextPlanStreamer(const Args & args, llama_model * model)
        : args_(args), model_(model) {
        load_geometry();
        open_source();
        initialize_gpu_slot();
        if (uses_full_msi(args_)) {
            initialize_plan();
        }
        initialize_cache();
    }

    Qwen3NextPlanStreamer(const Qwen3NextPlanStreamer &) = delete;
    Qwen3NextPlanStreamer & operator=(const Qwen3NextPlanStreamer &) = delete;

    ~Qwen3NextPlanStreamer() {
        if (plan_ && !shutdown_) {
            try {
                if (step_active_) {
                    plan_->reset(0);
                }
                plan_->shutdown();
            } catch (...) {
            }
        }
        if (cache_data_ != nullptr) {
            cudaFree(cache_data_);
        }
        if (plan_buffer_ != nullptr) {
            ggml_backend_buffer_free(plan_buffer_);
        }
        if (plan_ctx_ != nullptr) {
            ggml_free(plan_ctx_);
        }
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
        }
        source_service_.reset();
        if (host_windows_registered_ && host_arena_ != nullptr) {
            cudaHostUnregister(host_arena_);
            host_windows_registered_ = false;
        }
        if (host_arena_ != nullptr) {
            ::free(host_arena_);
            host_arena_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    static bool callback(ggml_tensor * tensor, bool ask, void * user_data) {
        auto * self = static_cast<Qwen3NextPlanStreamer *>(user_data);
        int layer = -1;
        if (sscanf(ggml_get_name(tensor), "ffn_moe_topk-%d", &layer) != 1 ||
            layer < 0 || layer >= kLayers) {
            return false;
        }
        if (ask) {
            return true;
        }
        try {
            self->service_layer(layer, tensor);
            return true;
        } catch (const std::exception & error) {
            self->callback_error_ = error.what();
            return false;
        }
    }

    void begin_step(std::uint64_t request_id) {
        require(!step_active_, "streamed step already active");
        callback_error_.clear();
        layers_seen_.clear();
        current_chunk_ = 0;
        if (plan_) {
            plan_->begin_request(++plan_request_sequence_);
        }
        if (source_service_) {
            step_source_begin_ = source_service_->snapshot();
            step_source_snapshot_ = true;
        }
        step_active_ = true;
        current_step_ = request_id - 1;
    }

    void finish_step() {
        require(step_active_, "streamed step is not active");
        require(callback_error_.empty(), "route callback failed: " + callback_error_);
        require(layers_seen_.size() == kLayers, "stock graph did not visit all Qwen3-Next MoE layers");
        for (int layer = 0; layer < kLayers; ++layer) {
            require(layers_seen_.count(layer) == 1, "stock graph skipped a Qwen3-Next MoE layer");
        }
        if (source_service_) {
            require(step_source_snapshot_, "source phase snapshot is missing");
            const auto end = source_service_->snapshot();
            PhaseSource & phase = current_step_ == 0 ? prefill_source_ : decode_source_;
            phase.logical_bytes += end.logical_bytes - step_source_begin_.logical_bytes;
            phase.physical_read_bytes +=
                end.physical_read_bytes - step_source_begin_.physical_read_bytes;
            phase.read_wall_ns += end.read_wall_ns - step_source_begin_.read_wall_ns;
            phase.exposed_wait_ns +=
                end.exposed_wait_ns - step_source_begin_.exposed_wait_ns;
            phase.h2d_completed_bytes +=
                end.h2d_completed_bytes - step_source_begin_.h2d_completed_bytes;
            step_source_snapshot_ = false;
        }
        if (plan_) {
            plan_->finish_request();
        }
        step_active_ = false;
    }

    void abort_step() {
        if (step_active_) {
            if (plan_) {
                plan_->reset(0);
            }
            step_active_ = false;
            step_source_snapshot_ = false;
        }
    }

    bool fatal_fault_triggered() const {
        return fault_triggered_ && fault_fatal_;
    }

    Json fault_report() const {
        return {
            {"requested", args_.fault},
            {"triggered", fault_triggered_},
            {"fatal", fault_fatal_},
            {"stage", fault_stage_},
            {"rejection", fault_rejection_},
            {"details", fault_details_},
            {"terminal_plan_snapshot", fault_terminal_snapshot_},
            {"terminal_source_snapshot", fault_terminal_source_},
        };
    }

    void shutdown() {
        require(!step_active_, "cannot shutdown with an active decode step");
        if (plan_) {
            plan_->reset(0);
            pre_shutdown_ = snapshot_json(plan_->snapshot());
            plan_->shutdown();
            shutdown_snapshot_ = snapshot_json(plan_->snapshot());
        } else {
            pre_shutdown_ = {{"lifecycle", "minimal_synchronous"},
                             {"live_consumers", 0},
                             {"active_operations", 0}};
            shutdown_snapshot_ = {{"lifecycle", "shutdown"},
                                  {"live_consumers", 0},
                                  {"active_operations", 0}};
        }
        if (source_service_) {
            source_service_->shutdown();
        }
        shutdown_ = true;
    }

    Json report() const {
        Json occupancy = Json::array();
        Json recycled_slots = Json::array();
        Json populations_per_layer = Json::array();
        Json decode_populations_per_layer = Json::array();
        Json final_cache_state = Json::array();
        std::uint64_t recycled_slot_total = 0;
        for (std::size_t layer_index = 0; layer_index < cache_entries_.size();
             ++layer_index) {
            const auto & layer = cache_entries_[layer_index];
            std::uint64_t count = 0;
            std::uint64_t recycled = 0;
            Json layer_state = Json::array();
            for (const auto & entry : layer) {
                count += entry.valid ? 1 : 0;
                recycled += entry.recycled_during_decode ? 1 : 0;
                layer_state.push_back({
                    {"expert", entry.expert},
                    {"generation", entry.generation},
                    {"valid", entry.valid},
                    {"recycled_during_decode", entry.recycled_during_decode},
                });
            }
            occupancy.push_back(count);
            recycled_slots.push_back(recycled);
            recycled_slot_total += recycled;
            populations_per_layer.push_back(cache_populations_per_layer_[layer_index]);
            decode_populations_per_layer.push_back(
                decode_cache_populations_per_layer_[layer_index]);
            final_cache_state.push_back(std::move(layer_state));
        }
        const std::uint64_t cache_miss_bytes = materialized_bytes_;
        const std::uint64_t cache_hit_bytes = cache_hit_bytes_;
        return {
            {"component_manifest_valid", true},
            {"source_mode", source_mode_name(source_mode(args_))},
            {"lifecycle_profile", plan_ ? "full_msi" : "minimal_synchronous"},
            {"layers", kLayers},
            {"experts", kExperts},
            {"experts_used", kExpertsUsed},
            {"expert_bundle_bytes", bundle_bytes_},
            {"source_object_identity", {
                {"expert_tensor_count", kLayers * 3},
                {"gate_up", {
                    {"native_qtype", "Q4_K"},
                    {"tensor_count", kLayers * 2},
                    {"expert_bytes", gate_bytes_},
                }},
                {"down", {
                    {"q4_k_tensor_count", down_q4_layers_},
                    {"q6_k_tensor_count", down_q6_layers_},
                    {"q4_k_expert_bytes", 589824},
                    {"q6_k_expert_bytes", 860160},
                }},
                {"execution_time_conversion", false},
            }},
            {"shared_destination_raw_bytes", shared_destination_raw_bytes_},
            {"plan_gpu_slot_bytes", plan_buffer_bytes_},
            {"host_window_bytes", host_arena_bytes_},
            {"bounded_source", source_report()},
            {"fault", fault_report()},
            {"phase_source", {
                {"prefill", phase_source_json(prefill_source_)},
                {"decode", phase_source_json(decode_source_)},
            }},
            {"materializations", materializations_},
            {"unique_selected_expert_instances", cache_enabled_ ? cache_accesses_ : materializations_},
            {"storage_bytes", materialized_bytes_},
            {"source_cache_advice", {
                {"enabled", args_.drop_source_cache},
                {"calls", source_cache_advice_calls_},
                {"bytes", source_cache_advice_bytes_},
            }},
            {"h2d_bytes", materialized_bytes_},
            {"d2d_scatter_bytes", materialized_bytes_ + cache_hit_bytes_},
            {"slot_reuse_count", materializations_ == 0 ? 0 : materializations_ - 1},
            {"window_reuse_count", materializations_ == 0 ? 0 : materializations_ - 1},
            {"cache", {
                {"enabled", cache_enabled_},
                {"policy", cache_enabled_ ? "per_layer_lru" : "disabled"},
                {"capacity_per_layer", args_.cache_capacity},
                {"requested_bytes", cache_bytes_},
                {"accesses", cache_accesses_},
                {"hits", cache_hits_},
                {"misses", cache_misses_},
                {"hit_rate", cache_accesses_ == 0 ? 0.0 :
                    static_cast<double>(cache_hits_) / static_cast<double>(cache_accesses_)},
                {"prefill_accesses", prefill_cache_accesses_},
                {"prefill_hits", prefill_cache_hits_},
                {"prefill_misses", prefill_cache_misses_},
                {"decode_accesses", decode_cache_accesses_},
                {"decode_hits", decode_cache_hits_},
                {"decode_misses", decode_cache_misses_},
                {"logical_hit_bytes", cache_hit_bytes},
                {"logical_miss_bytes", cache_miss_bytes},
                {"populations", cache_populations_},
                {"populations_per_layer", populations_per_layer},
                {"decode_populations", decode_cache_populations_},
                {"decode_populations_per_layer", decode_populations_per_layer},
                {"evictions", cache_evictions_},
                {"decode_recycled_slots", recycled_slot_total},
                {"decode_recycled_slots_per_layer", recycled_slots},
                {"decode_recycle_events_per_layer", decode_cache_recycles_per_layer_},
                {"prefill_tail_population_skips", prefill_tail_population_skips_},
                {"generation_updates", cache_generation_},
                {"stale_entry_rejections", cache_stale_rejections_},
                {"premature_reuse_rejections", cache_premature_reuse_rejections_},
                {"cache_fill_d2d_bytes", cache_populations_ * bundle_bytes_},
                {"cache_hit_d2d_bytes", cache_hit_bytes},
                {"destination_d2d_bytes", (materializations_ + cache_hits_) * bundle_bytes_},
                {"final_occupancy_per_layer", occupancy},
                {"max_occupancy_per_layer", max_cache_occupancy_},
                {"final_state", final_cache_state},
            }},
            {"timing_ns", {
                {"storage", timings_.source_ns},
                {"h2d", timings_.h2d_ns},
                {"d2d_scatter", timings_.scatter_ns},
                {"cache_fill", timings_.cache_fill_ns},
            }},
            {"route_events", route_events_},
            {"pre_shutdown_snapshot", pre_shutdown_},
            {"shutdown_snapshot", shutdown_snapshot_},
        };
    }

  private:
    void record_fault(const std::string & stage, bool fatal,
                      const std::string & rejection, Json details = Json::object()) {
        require(!fault_triggered_, "fault hook triggered more than once");
        fault_triggered_ = true;
        fault_fatal_ = fatal;
        fault_stage_ = stage;
        fault_rejection_ = rejection;
        fault_details_ = std::move(details);
        fault_terminal_snapshot_ = snapshot_json(plan_->snapshot());
        fault_terminal_source_ = source_report();
    }

    template <typename Action>
    void expect_nonfatal_rejection(const std::string & stage, Action action,
                                   Json details = Json::object()) {
        const Json before = snapshot_json(plan_->snapshot());
        try {
            action();
        } catch (const std::exception & error) {
            details["before"] = before;
            details["after"] = snapshot_json(plan_->snapshot());
            record_fault(stage, false, error.what(), std::move(details));
            return;
        }
        throw std::runtime_error("fault hook was unexpectedly accepted: " + stage);
    }

    struct CacheEntry {
        int expert = -1;
        std::uint64_t last_use = 0;
        std::uint64_t generation = 0;
        bool valid = false;
        bool recycled_during_decode = false;
    };

    struct PendingSource {
        SourceTicket ticket;
        std::size_t window_index = 0;
    };

    static TensorGeometry source_geometry(const gguf_context * gguf, int layer,
                                          const char * suffix,
                                          ggml_tensor * destination) {
        const std::string name = tensor_name(layer, suffix);
        const int64_t tensor_id = gguf_find_tensor(gguf, name.c_str());
        require(tensor_id >= 0, "GGUF tensor is missing: " + name);
        const std::uint64_t tensor_bytes = gguf_get_tensor_size(gguf, tensor_id);
        require(tensor_bytes % kExperts == 0, "expert tensor does not split evenly: " + name);
        require(destination != nullptr, "runtime tensor is missing: " + name);
        require(ggml_nbytes(destination) == tensor_bytes,
                "runtime/GGUF tensor size differs: " + name);
        return {
            gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tensor_id),
            tensor_bytes,
            tensor_bytes / kExperts,
            gguf_get_tensor_type(gguf, tensor_id),
            destination,
        };
    }

    void load_geometry() {
        std::unordered_map<std::string, ggml_tensor *> tensors;
        for (const auto & [name, tensor] : llama_internal_get_tensor_map(model_)) {
            tensors.emplace(name, tensor);
        }
        ggml_context * metadata_ctx = nullptr;
        gguf_init_params params{true, &metadata_ctx};
        gguf_context * gguf = gguf_init_from_file(args_.model.c_str(), params);
        require(gguf != nullptr && metadata_ctx != nullptr, "cannot open GGUF metadata");
        try {
            for (int layer = 0; layer < kLayers; ++layer) {
                layers_[layer].gate = source_geometry(
                    gguf, layer, "ffn_gate_exps.weight",
                    tensors.at(tensor_name(layer, "ffn_gate_exps.weight")));
                layers_[layer].up = source_geometry(
                    gguf, layer, "ffn_up_exps.weight",
                    tensors.at(tensor_name(layer, "ffn_up_exps.weight")));
                layers_[layer].down = source_geometry(
                    gguf, layer, "ffn_down_exps.weight",
                    tensors.at(tensor_name(layer, "ffn_down_exps.weight")));
            }
            gate_bytes_ = layers_[0].gate.expert_bytes;
            up_bytes_ = layers_[0].up.expert_bytes;
            down_bytes_ = 0;
            down_q4_layers_ = 0;
            down_q6_layers_ = 0;
            managed_expert_inventory_bytes_ = 0;
            for (const auto & layer : layers_) {
                require(layer.gate.expert_bytes == gate_bytes_ &&
                        layer.up.expert_bytes == up_bytes_ &&
                        layer.gate.native_type == GGML_TYPE_Q4_K &&
                        layer.up.native_type == GGML_TYPE_Q4_K,
                        "Qwen3-Next gate/up expert geometry is not uniform");
                if (layer.down.native_type == GGML_TYPE_Q4_K) {
                    require(layer.down.expert_bytes == 589824,
                            "Qwen3-Next Q4_K down expert geometry changed");
                    ++down_q4_layers_;
                } else if (layer.down.native_type == GGML_TYPE_Q6_K) {
                    require(layer.down.expert_bytes == 860160,
                            "Qwen3-Next Q6_K down expert geometry changed");
                    ++down_q6_layers_;
                } else {
                    throw std::runtime_error("Qwen3-Next down expert native qtype changed");
                }
                down_bytes_ = std::max(down_bytes_, layer.down.expert_bytes);
                managed_expert_inventory_bytes_ +=
                    layer.gate.tensor_bytes + layer.up.tensor_bytes + layer.down.tensor_bytes;
            }
            bundle_bytes_ = gate_bytes_ + up_bytes_ + down_bytes_;
            source_file_bytes_ = file_size(args_.model);
            require(gate_bytes_ == 589824 && up_bytes_ == 589824 && down_bytes_ == 860160,
                    "official Qwen3-Next expert geometry changed");
            require(down_q4_layers_ == 24 && down_q6_layers_ == 24,
                    "official Qwen3-Next down-expert qtype population changed");
            require(source_file_bytes_ == kOfficialModelBytes,
                    "official Qwen3-Next source-file size changed");
            require(managed_expert_inventory_bytes_ == 46808432640ULL,
                    "official Qwen3-Next managed expert inventory changed");
            for (const auto & layer : layers_) {
                require(layer.gate.destination->data == layers_[0].gate.destination->data &&
                        layer.up.destination->data == layers_[0].up.destination->data &&
                        layer.down.destination->data == layers_[0].down.destination->data,
                        "runtime expert descriptors do not share the streamed destination");
            }
            shared_destination_raw_bytes_ = gate_bytes_ * kExperts +
                                            up_bytes_ * kExperts +
                                            down_bytes_ * kExperts;
        } catch (...) {
            gguf_free(gguf);
            ggml_free(metadata_ctx);
            throw;
        }
        gguf_free(gguf);
        ggml_free(metadata_ctx);
    }

    std::uint64_t layer_bundle_bytes(int layer) const {
        return layers_[layer].gate.expert_bytes +
               layers_[layer].up.expert_bytes +
               layers_[layer].down.expert_bytes;
    }

    void open_source() {
        host_window_count_ = args_.bounded_source ?
            static_cast<std::size_t>(args_.source_in_flight) : 1;
        const auto round_up = [](std::uint64_t value, std::uint64_t alignment) {
            return (value + alignment - 1) & ~(alignment - 1);
        };
        direct_scratch_offset_ = round_up(bundle_bytes_, 4096);
        direct_scratch_bytes_ = round_up(down_bytes_ + 4095, 4096);
        host_window_stride_ = direct_scratch_offset_ + direct_scratch_bytes_;
        host_arena_bytes_ = host_window_count_ * host_window_stride_;
        const int allocation = ::posix_memalign(&host_arena_, 4096, host_arena_bytes_);
        require(allocation == 0 && host_arena_ != nullptr,
                "cannot allocate fixed aligned source windows");
        std::memset(host_arena_, 0, host_arena_bytes_);
        const cudaError_t register_status = cudaHostRegister(
            host_arena_, host_arena_bytes_, cudaHostRegisterPortable);
        require(register_status == cudaSuccess,
                std::string("source host-window registration failed: ") +
                cudaGetErrorString(register_status));
        host_windows_registered_ = true;
        if (!args_.bounded_source) {
            int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_DIRECT
            if (source_mode(args_) == SourceMode::MinimalDirect) {
                flags |= O_DIRECT;
            }
#else
            if (source_mode(args_) == SourceMode::MinimalDirect) {
                throw std::runtime_error("O_DIRECT is unavailable on this platform");
            }
#endif
            fd_ = open(args_.model.c_str(), flags);
            require(fd_ >= 0,
                    "cannot open GGUF payload: " + std::string(std::strerror(errno)));
            return;
        }
        std::vector<SourceWindow> windows;
        windows.reserve(host_window_count_);
        for (std::size_t index = 0; index < host_window_count_; ++index) {
            windows.push_back(SourceWindow{
                kWindowId + static_cast<Id>(index),
                address_of(host_window_pointer(index)),
                bundle_bytes_,
                address_of(direct_scratch_pointer(index)),
                direct_scratch_bytes_,
            });
        }
        source_service_ = std::make_unique<SourceService>(
            args_.model, std::move(windows), host_window_count_, args_.source_direct,
            4096, args_.drop_source_cache, source_fault_mode(args_));
    }

    char * host_window_pointer(std::size_t index) const {
        require(index < host_window_count_ && host_arena_ != nullptr,
                "source host-window index is invalid");
        return static_cast<char *>(host_arena_) + index * host_window_stride_;
    }

    char * direct_scratch_pointer(std::size_t index) const {
        return host_window_pointer(index) + direct_scratch_offset_;
    }

    Json source_report() const {
        const Json identity = {
            {"source_file_bytes", source_file_bytes_},
            {"managed_expert_inventory_bytes", managed_expert_inventory_bytes_},
        };
        if (!source_service_) {
            Json result = {
                {"enabled", source_mode(args_) != SourceMode::Legacy},
                {"mode", source_mode_name(source_mode(args_))},
                {"in_flight_limit", 1},
                {"direct_io", source_mode(args_) == SourceMode::MinimalDirect},
                {"registered_host_windows", host_windows_registered_},
                {"submissions", simple_source_.submissions},
                {"completions", simple_source_.completions},
                {"logical_bytes", simple_source_.logical_bytes},
                {"physical_read_bytes", simple_source_.physical_read_bytes},
                {"padding_bytes", simple_source_.padding_bytes},
                {"read_wall_ns", simple_source_.read_wall_ns},
                {"h2d_issued_bytes", simple_source_.h2d_issued_bytes},
                {"h2d_completed_bytes", simple_source_.h2d_completed_bytes},
                {"fixed_direct_reads", simple_source_.fixed_direct_reads},
                {"dynamic_direct_allocations", 0},
                {"peak_in_flight", simple_source_.submissions == 0 ? 0 : 1},
                {"active_tickets", 0},
                {"free_windows", host_window_count_},
            };
            result.update(identity);
            return result;
        }
        const auto value = source_service_->snapshot();
        Json result = {
            {"enabled", true},
            {"mode", source_mode_name(source_mode(args_))},
            {"in_flight_limit", source_service_->max_in_flight()},
            {"direct_io", source_service_->direct_io()},
            {"registered_host_windows", host_windows_registered_},
            {"submissions", value.submissions},
            {"completions", value.completions},
            {"cancellations", value.cancellations},
            {"failures", value.failures},
            {"logical_bytes", value.logical_bytes},
            {"physical_read_bytes", value.physical_read_bytes},
            {"block_read_bytes", value.block_read_bytes},
            {"padding_bytes", value.padding_bytes},
            {"read_wall_ns", value.read_wall_ns},
            {"exposed_wait_ns", value.exposed_wait_ns},
            {"h2d_issued_bytes", value.h2d_issued_bytes},
            {"h2d_completed_bytes", value.h2d_completed_bytes},
            {"queue_rejections", value.queue_rejections},
            {"lifecycle_rejections", value.lifecycle_rejections},
            {"generation_reuses", value.generation_reuses},
            {"fixed_direct_reads", value.fixed_direct_reads},
            {"dynamic_direct_allocations", value.dynamic_direct_allocations},
            {"injected_short_completions", value.injected_short_completions},
            {"injected_io_errors", value.injected_io_errors},
            {"injected_partial_bundles", value.injected_partial_bundles},
            {"injected_completed_extents", value.injected_completed_extents},
            {"peak_in_flight", value.peak_in_flight},
            {"active_tickets", value.active_tickets},
            {"free_windows", value.free_windows},
            {"filling_windows", value.filling_windows},
            {"ready_windows", value.ready_windows},
            {"copying_windows", value.copying_windows},
            {"retirable_windows", value.retirable_windows},
        };
        result.update(identity);
        return result;
    }

    void initialize_gpu_slot() {
        ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        require(device != nullptr, "GPU backend is unavailable");
        backend_ = ggml_backend_dev_init(device, nullptr);
        require(backend_ != nullptr, "GPU backend initialization failed");
        ggml_init_params params{1024 * 1024, nullptr, true};
        plan_ctx_ = ggml_init(params);
        require(plan_ctx_ != nullptr, "Plan tensor context initialization failed");
        plan_tensor_ = ggml_new_tensor_1d(plan_ctx_, GGML_TYPE_I8,
                                          static_cast<int64_t>(bundle_bytes_));
        plan_buffer_ = ggml_backend_alloc_ctx_tensors(plan_ctx_, backend_);
        require(plan_buffer_ != nullptr, "Plan GPU slot allocation failed");
        plan_buffer_bytes_ = ggml_backend_buffer_get_size(plan_buffer_);
    }

    Manifest make_component_manifest() const {
        Manifest manifest;
        manifest.manifest_identity = 0x5157454e334e5854ULL;
        const std::uint64_t gpu_base = address_of(plan_tensor_->data);
        manifest.arena = {1, gpu_base, plan_buffer_bytes_};
        // Plan owns only the logical service windows.  The aligned scratch tail
        // in each pinned stride is private O_DIRECT transport workspace and is
        // intentionally absent from the service-window capacity ledger.
        manifest.managed_host_bytes = host_window_count_ * bundle_bytes_;
        manifest.files = {
            FileManifest{kSourceFileId, file_size(args_.model)},
            FileManifest{kMaterializedFileId,
                         static_cast<std::uint64_t>(kLayers) * kExperts * bundle_bytes_},
        };
        manifest.gpu_slots = {{kSlotId, plan_buffer_bytes_, gpu_base}};
        std::vector<Id> allowed_window_ids;
        for (int index = 0; index < args_.source_in_flight; ++index) {
            const Id window_id = kWindowId + static_cast<Id>(index);
            allowed_window_ids.push_back(window_id);
            manifest.host_windows.push_back({
                window_id,
                address_of(host_window_pointer(static_cast<std::size_t>(index))),
                bundle_bytes_,
            });
        }
        manifest.bundles.reserve(kLayers * kExperts);
        for (int layer = 0; layer < kLayers; ++layer) {
            for (int expert = 0; expert < kExperts; ++expert) {
                const Id object = object_id(layer, expert);
                const std::uint64_t ordinal = static_cast<std::uint64_t>(layer * kExperts + expert);
                const std::uint64_t materialized_base = ordinal * bundle_bytes_;
                const Id component_base = 1000000 + ordinal * 3;
                const Id tensor_base = 2000000 + static_cast<Id>(layer) * 3;
                BundleManifest bundle;
                bundle.object_id = object;
                bundle.graph_role_id = 4000 + layer;
                bundle.canonical_artifact_id = 5000;
                bundle.transformation_id = 5001;
                bundle.bundle_identity = 6000000 + ordinal;
                bundle.execution_layout_id = 5002;
                bundle.backend_id = 5003;
                bundle.slot_id = msi::plan_service::kInvalidId;
                bundle.allowed_slot_ids = {kSlotId};
                bundle.host_window_id = msi::plan_service::kInvalidId;
                bundle.allowed_host_window_ids = allowed_window_ids;
                bundle.stable_address = gpu_base;
                bundle.materialized_bytes = layer_bundle_bytes(layer);
                bundle.slot_requirement_bytes = bundle_bytes_;
                bundle.window_requirement_bytes = bundle_bytes_;
                bundle.required_alignment = 128;
                bundle.max_consumers = 1;
                bundle.canonical_tensor_ids = {tensor_base, tensor_base + 1, tensor_base + 2};
                bundle.required_role_ids = {kGateRole, kUpRole, kDownRole};
                bundle.components = {
                    ComponentManifest{
                        component_base, kGateRole, tensor_base,
                        Extent{kSourceFileId,
                               layers_[layer].gate.source_offset + expert * gate_bytes_,
                               gate_bytes_, 32, component_base + 10000000},
                        Extent{kMaterializedFileId, materialized_base,
                               gate_bytes_, 1, component_base + 20000000},
                        DestinationView{component_base + 30000000, 0, gate_bytes_, 128},
                        144, true, {}},
                    ComponentManifest{
                        component_base + 1, kUpRole, tensor_base + 1,
                        Extent{kSourceFileId,
                               layers_[layer].up.source_offset + expert * layers_[layer].up.expert_bytes,
                               layers_[layer].up.expert_bytes, 32, component_base + 10000001},
                        Extent{kMaterializedFileId, materialized_base + gate_bytes_,
                               layers_[layer].up.expert_bytes, 1, component_base + 20000001},
                        DestinationView{component_base + 30000001, gate_bytes_,
                                        layers_[layer].up.expert_bytes, 128},
                        144, true, {}},
                    ComponentManifest{
                        component_base + 2, kDownRole, tensor_base + 2,
                        Extent{kSourceFileId,
                               layers_[layer].down.source_offset + expert * layers_[layer].down.expert_bytes,
                               layers_[layer].down.expert_bytes, 32, component_base + 10000002},
                        Extent{kMaterializedFileId, materialized_base + gate_bytes_ + up_bytes_,
                               layers_[layer].down.expert_bytes, 1, component_base + 20000002},
                        DestinationView{component_base + 30000002,
                                        gate_bytes_ + up_bytes_,
                                        layers_[layer].down.expert_bytes, 128},
                        layers_[layer].down.expert_bytes == 589824 ? 144ULL : 210ULL,
                        true, {}},
                };
                manifest.bundles.push_back(std::move(bundle));
            }
        }
        return manifest;
    }

    void initialize_plan() {
        Manifest manifest = make_component_manifest();
        msi::plan_service::validate_manifest(manifest);
        plan_ = std::make_unique<Engine>(msi::plan_service::lower_to_plan_v0(manifest));
        plan_->initialize();
    }

    void initialize_cache() {
        cache_enabled_ = args_.cache_capacity > 0;
        if (!cache_enabled_) {
            return;
        }
        cache_bytes_ = static_cast<std::uint64_t>(kLayers) *
                       static_cast<std::uint64_t>(args_.cache_capacity) * bundle_bytes_;
        const cudaError_t status = cudaMalloc(&cache_data_, cache_bytes_);
        require(status == cudaSuccess,
                std::string("CUDA compact cache allocation failed: ") +
                cudaGetErrorString(status));
        for (auto & layer : cache_entries_) {
            layer.resize(static_cast<std::size_t>(args_.cache_capacity));
        }
        max_cache_occupancy_.assign(kLayers, 0);
    }

    static void copy_d2d(void * destination, const void * source, std::size_t bytes) {
        const cudaError_t status = cudaMemcpyAsync(destination, source, bytes,
                                                   cudaMemcpyDeviceToDevice,
                                                   cudaStreamPerThread);
        require(status == cudaSuccess,
                std::string("CUDA D2D copy failed: ") + cudaGetErrorString(status));
    }

    static void synchronize_d2d() {
        const cudaError_t status = cudaStreamSynchronize(cudaStreamPerThread);
        require(status == cudaSuccess,
                std::string("CUDA D2D synchronization failed: ") +
                cudaGetErrorString(status));
    }

    void copy_to_destination(int layer, int expert, const char * source) {
        const auto begin = Clock::now();
        copy_d2d(static_cast<char *>(layers_[layer].gate.destination->data) +
                     expert * layers_[layer].gate.expert_bytes,
                 source, layers_[layer].gate.expert_bytes);
        copy_d2d(static_cast<char *>(layers_[layer].up.destination->data) +
                     expert * layers_[layer].up.expert_bytes,
                 source + gate_bytes_, layers_[layer].up.expert_bytes);
        copy_d2d(static_cast<char *>(layers_[layer].down.destination->data) +
                     expert * layers_[layer].down.expert_bytes,
                 source + gate_bytes_ + up_bytes_, layers_[layer].down.expert_bytes);
        synchronize_d2d();
        timings_.scatter_ns += elapsed_ns(begin, Clock::now());
    }

    char * cache_slot_pointer(int layer, std::size_t slot) const {
        require(cache_data_ != nullptr && slot < cache_entries_[layer].size(),
                "compact cache slot address is invalid");
        const std::uint64_t ordinal =
            static_cast<std::uint64_t>(layer) * args_.cache_capacity + slot;
        return static_cast<char *>(cache_data_) + ordinal * bundle_bytes_;
    }

    std::size_t find_cache_entry(int layer, int expert) const {
        const auto & entries = cache_entries_[layer];
        for (std::size_t index = 0; index < entries.size(); ++index) {
            if (entries[index].valid && entries[index].expert == expert) {
                return index;
            }
        }
        return entries.size();
    }

    std::size_t choose_cache_victim(int layer) const {
        const auto & entries = cache_entries_[layer];
        require(!entries.empty(), "compact cache has no entries");
        for (std::size_t index = 0; index < entries.size(); ++index) {
            if (!entries[index].valid) {
                return index;
            }
        }
        return static_cast<std::size_t>(std::distance(
            entries.begin(),
            std::min_element(entries.begin(), entries.end(),
                             [](const CacheEntry & left, const CacheEntry & right) {
                                 return left.last_use < right.last_use;
                             })));
    }

    static std::size_t choose_cache_victim(
            const std::vector<CacheEntry> & entries) {
        require(!entries.empty(), "compact cache simulation has no entries");
        for (std::size_t index = 0; index < entries.size(); ++index) {
            if (!entries[index].valid) {
                return index;
            }
        }
        return static_cast<std::size_t>(std::distance(
            entries.begin(),
            std::min_element(entries.begin(), entries.end(),
                             [](const CacheEntry & left, const CacheEntry & right) {
                                 return left.last_use < right.last_use;
                             })));
    }

    std::vector<SourceExtent> source_extents(int layer, int expert) const {
        return {
            SourceExtent{layers_[layer].gate.source_offset + expert * layers_[layer].gate.expert_bytes,
                         layers_[layer].gate.expert_bytes, 0},
            SourceExtent{layers_[layer].up.source_offset + expert * layers_[layer].up.expert_bytes,
                         layers_[layer].up.expert_bytes, gate_bytes_},
            SourceExtent{layers_[layer].down.source_offset + expert * layers_[layer].down.expert_bytes,
                         layers_[layer].down.expert_bytes, gate_bytes_ + up_bytes_},
        };
    }

    void drain_unstarted_source_tasks() {
        for (const auto & [expert, pending] : pending_source_) {
            (void)expert;
            source_service_->await(pending.ticket);
            source_service_->mark_retirable(pending.ticket);
            source_service_->retire(pending.ticket);
        }
        pending_source_.clear();
    }

    void read_synchronous_source(int layer, int expert) {
        require(!source_service_ && fd_ >= 0,
                "synchronous source identity is unavailable");
        const auto begin = Clock::now();
        std::uint64_t physical_bytes = 0;
        std::uint64_t padding_bytes = 0;
        std::uint64_t fixed_direct_reads = 0;
        for (const SourceExtent & extent : source_extents(layer, expert)) {
            char * destination = host_window_pointer(0) + extent.destination_offset;
            if (source_mode(args_) != SourceMode::MinimalDirect) {
                read_exact(fd_, extent.source_offset, destination,
                           static_cast<std::size_t>(extent.bytes));
                if (source_mode(args_) == SourceMode::OracleBuffered) {
                    const int status = posix_fadvise(
                        fd_, static_cast<off_t>(extent.source_offset),
                        static_cast<off_t>(extent.bytes), POSIX_FADV_DONTNEED);
                    require(status == 0, "oracle source cache discard failed: " +
                                             std::string(std::strerror(status)));
                    ++source_cache_advice_calls_;
                    source_cache_advice_bytes_ += extent.bytes;
                }
                continue;
            }
            const std::uint64_t aligned_begin = extent.source_offset & ~4095ULL;
            const std::uint64_t aligned_end =
                (extent.source_offset + extent.bytes + 4095ULL) & ~4095ULL;
            const std::uint64_t envelope_bytes = aligned_end - aligned_begin;
            require(envelope_bytes <= direct_scratch_bytes_,
                    "minimal direct envelope exceeds its fixed window");
            char * scratch = direct_scratch_pointer(0);
            read_exact(fd_, aligned_begin, scratch,
                       static_cast<std::size_t>(envelope_bytes));
            std::memcpy(destination,
                        scratch + (extent.source_offset - aligned_begin),
                        static_cast<std::size_t>(extent.bytes));
            physical_bytes += envelope_bytes;
            padding_bytes += envelope_bytes - extent.bytes;
            ++fixed_direct_reads;
        }
        const std::uint64_t read_ns = elapsed_ns(begin, Clock::now());
        const std::uint64_t logical_bytes = layer_bundle_bytes(layer);
        ++simple_source_.submissions;
        ++simple_source_.completions;
        simple_source_.logical_bytes += logical_bytes;
        simple_source_.physical_read_bytes += physical_bytes;
        simple_source_.padding_bytes += padding_bytes;
        simple_source_.read_wall_ns += read_ns;
        simple_source_.fixed_direct_reads += fixed_direct_reads;
        timings_.source_ns += read_ns;
        PhaseSource & phase = current_step_ == 0 ? prefill_source_ : decode_source_;
        phase.logical_bytes += logical_bytes;
        phase.physical_read_bytes += physical_bytes;
        phase.read_wall_ns += read_ns;
        phase.exposed_wait_ns += read_ns;
    }

    void submit_chunk(int layer, const std::vector<int> & experts,
                      const std::vector<bool> & populate) {
        require(source_service_ && pending_source_.empty() &&
                experts.size() == populate.size(),
                "invalid bounded source chunk submission");
        std::vector<bool> misses(experts.size(), false);
        if (!cache_enabled_) {
            std::fill(misses.begin(), misses.end(), true);
        } else {
            std::vector<CacheEntry> simulated = cache_entries_[layer];
            std::uint64_t simulated_clock = cache_clock_;
            for (std::size_t ordinal = 0; ordinal < experts.size(); ++ordinal) {
                ++simulated_clock;
                const int expert = experts[ordinal];
                auto found = std::find_if(
                    simulated.begin(), simulated.end(),
                    [expert](const CacheEntry & entry) {
                        return entry.valid && entry.expert == expert;
                    });
                if (found != simulated.end()) {
                    found->last_use = simulated_clock;
                    continue;
                }
                misses[ordinal] = true;
                if (populate[ordinal]) {
                    CacheEntry & victim = simulated[choose_cache_victim(simulated)];
                    victim = CacheEntry{expert, simulated_clock, 1, true};
                }
            }
        }
        std::size_t window_index = 0;
        for (std::size_t ordinal = 0; ordinal < experts.size(); ++ordinal) {
            if (!misses[ordinal]) {
                continue;
            }
            require(window_index < static_cast<std::size_t>(args_.source_in_flight),
                    "bounded source chunk exceeds its fixed windows");
            const int expert = experts[ordinal];
            const Id window_id = kWindowId + static_cast<Id>(window_index);
            PendingSource pending{
                source_service_->submit(window_id, source_extents(layer, expert)),
                window_index,
            };
            require(pending_source_.emplace(expert, pending).second,
                    "bounded source expert was submitted twice");
            ++window_index;
        }
    }

    void update_cache_occupancy(int layer) {
        std::uint64_t occupancy = 0;
        for (const auto & entry : cache_entries_[layer]) {
            occupancy += entry.valid ? 1 : 0;
        }
        max_cache_occupancy_[layer] = std::max(max_cache_occupancy_[layer], occupancy);
    }

    void record_cache_population(int layer, CacheEntry & entry, int expert,
                                 bool evicted) {
        if (evicted && current_step_ > 0) {
            entry.recycled_during_decode = true;
            ++decode_cache_recycles_per_layer_[layer];
        }
        entry.expert = expert;
        entry.last_use = cache_clock_;
        entry.generation = ++cache_generation_;
        entry.valid = true;
        ++cache_populations_;
        ++cache_populations_per_layer_[layer];
        cache_evictions_ += evicted ? 1 : 0;
        if (current_step_ > 0) {
            ++decode_cache_populations_;
            ++decode_cache_populations_per_layer_[layer];
        }
        update_cache_occupancy(layer);
    }

    void service_layer(int layer, ggml_tensor * selected_tensor) {
        require(step_active_, "route callback executed outside a Plan decode step");
        if (layers_seen_.count(layer) != 0) {
            require(layers_seen_.size() == kLayers,
                    "stock graph repeated a layer before completing a Qwen3-Next chunk");
            if (plan_) {
                plan_->finish_request();
            }
            layers_seen_.clear();
            ++current_chunk_;
            if (plan_) {
                plan_->begin_request(++plan_request_sequence_);
            }
        }
        require(layers_seen_.insert(layer).second,
                "Qwen3-Next graph chunk layer binding failed");
        const std::size_t selected_count = static_cast<std::size_t>(ggml_nelements(selected_tensor));
        require(selected_count > 0 && selected_count % kExpertsUsed == 0,
                "unexpected Qwen3-Next top-k tensor shape");
        std::vector<std::int32_t> selected(selected_count);
        ggml_backend_tensor_get(selected_tensor, selected.data(), 0,
                                selected.size() * sizeof(selected.front()));
        std::vector<int> unique;
        std::array<bool, kExperts> seen{};
        for (const std::int32_t expert : selected) {
            require(expert >= 0 && expert < kExperts, "router selected an invalid expert");
            if (!seen[expert]) {
                seen[expert] = true;
                unique.push_back(expert);
            }
        }
        Json event = {
            {"step", current_step_},
            {"chunk", current_chunk_},
            {"layer", layer},
            {"tokens", selected_count / kExpertsUsed},
            {"selected_unique", unique},
        };
        route_events_.push_back(std::move(event));
        const bool tail_seed = cache_enabled_ && current_step_ == 0 && current_chunk_ == 0 &&
                               unique.size() > static_cast<std::size_t>(args_.cache_capacity);
        if (tail_seed) {
            for (const auto & entry : cache_entries_[layer]) {
                require(!entry.valid,
                        "prefill-tail seeding requires an initially empty per-layer cache");
            }
        }
        const std::size_t population_begin = tail_seed ?
            unique.size() - static_cast<std::size_t>(args_.cache_capacity) : 0;
        const std::size_t chunk_size = args_.bounded_source ?
            static_cast<std::size_t>(args_.source_in_flight) : 1;
        for (std::size_t chunk_begin = 0; chunk_begin < unique.size();
             chunk_begin += chunk_size) {
            const std::size_t chunk_end = std::min(unique.size(), chunk_begin + chunk_size);
            std::vector<int> experts(unique.begin() + chunk_begin,
                                     unique.begin() + chunk_end);
            std::vector<bool> populate;
            populate.reserve(experts.size());
            for (std::size_t index = chunk_begin; index < chunk_end; ++index) {
                populate.push_back(!tail_seed || index >= population_begin);
            }
            if (source_service_) {
                submit_chunk(layer, experts, populate);
            }
            std::vector<std::size_t> service_order(experts.size());
            for (std::size_t index = 0; index < service_order.size(); ++index) {
                service_order[index] = index;
            }
            const bool inject_out_of_order =
                args_.fault == "out_of_order" && !fault_triggered_ &&
                experts.size() >= 2;
            Json out_of_order_details = Json::object();
            if (inject_out_of_order) {
                std::swap(service_order[0], service_order[1]);
                out_of_order_details = {
                    {"submission_ticket_ids", {
                        pending_source_.at(experts[0]).ticket.ticket_id,
                        pending_source_.at(experts[1]).ticket.ticket_id,
                    }},
                    {"completion_ticket_ids", {
                        pending_source_.at(experts[1]).ticket.ticket_id,
                        pending_source_.at(experts[0]).ticket.ticket_id,
                    }},
                };
            }
            for (const std::size_t index : service_order) {
                service_expert(layer, experts[index], populate[index]);
            }
            if (inject_out_of_order) {
                record_fault("out_of_order_valid_completions", false, "accepted",
                             std::move(out_of_order_details));
            }
            require(pending_source_.empty(),
                    "bounded source chunk ended with unconsumed reads");
        }
    }

    void service_expert(int layer, int expert, bool populate_on_miss) {
        if (!cache_enabled_) {
            materialize(layer, expert, false, 0);
            return;
        }
        ++cache_clock_;
        ++cache_accesses_;
        if (current_step_ == 0) {
            ++prefill_cache_accesses_;
        } else {
            ++decode_cache_accesses_;
        }
        const std::size_t hit_slot = find_cache_entry(layer, expert);
        if (hit_slot != cache_entries_[layer].size()) {
            CacheEntry & entry = cache_entries_[layer][hit_slot];
            require(entry.valid && entry.expert == expert && entry.generation > 0,
                    "compact cache entry identity/generation is stale");
            copy_to_destination(layer, expert, cache_slot_pointer(layer, hit_slot));
            entry.last_use = cache_clock_;
            ++cache_hits_;
            cache_hit_bytes_ += layer_bundle_bytes(layer);
            if (current_step_ == 0) {
                ++prefill_cache_hits_;
            } else {
                ++decode_cache_hits_;
            }
            return;
        }
        ++cache_misses_;
        if (current_step_ == 0) {
            ++prefill_cache_misses_;
        } else {
            ++decode_cache_misses_;
        }
        if (!populate_on_miss) {
            ++prefill_tail_population_skips_;
        }
        const std::size_t victim = populate_on_miss ? choose_cache_victim(layer) : 0;
        materialize(layer, expert, populate_on_miss, victim);
    }

    void materialize(int layer, int expert, bool populate_cache, std::size_t cache_slot) {
        if (!plan_) {
            materialize_synchronous(layer, expert, populate_cache, cache_slot);
            return;
        }
        const Id object = object_id(layer, expert);
        plan_->declare_lifetime(object, 0, 0);
        std::size_t window_index = 0;
        SourceTicket source_ticket;
        if (source_service_) {
            const auto pending = pending_source_.find(expert);
            require(pending != pending_source_.end(),
                    "bounded source miss has no submitted read");
            source_ticket = pending->second.ticket;
            window_index = pending->second.window_index;
        }
        const Id window_id = kWindowId + static_cast<Id>(window_index);
        const auto window = plan_->start_read(object, window_id);
        if (args_.fault == "reset_outstanding" && !fault_triggered_) {
            expect_nonfatal_rejection(
                "reset_with_source_read_outstanding",
                [this] { plan_->reset(0); },
                {{"window_id", window.window_id},
                 {"window_generation", window.generation}});
        }
        auto begin = Clock::now();
        if (source_service_) {
            const Json before = snapshot_json(plan_->snapshot());
            try {
                source_service_->await(source_ticket);
            } catch (const std::exception & error) {
                const bool expected = !fault_triggered_ &&
                    (args_.fault == "short_read" || args_.fault == "eio" ||
                     args_.fault == "partial_bundle");
                if (!expected) {
                    throw;
                }
                source_service_->retire(source_ticket);
                pending_source_.erase(expert);
                drain_unstarted_source_tasks();
                record_fault(
                    args_.fault == "short_read" ? "short_successful_read" :
                    args_.fault == "eio" ? "negative_eio_completion" :
                                            "failed_extent_in_bundle",
                    true, error.what(),
                    {{"before", before},
                     {"after", snapshot_json(plan_->snapshot())},
                     {"h2d_started", false},
                     {"host_ready_published", false}});
                throw std::runtime_error("injected fatal source fault: " + args_.fault);
            }
            pending_source_.erase(expert);
            if (args_.fault == "stale_completion" && !fault_triggered_ &&
                stale_window_.has_value() &&
                stale_window_->window_id == window.window_id &&
                stale_window_->generation < window.generation) {
                const Json stale_before = snapshot_json(plan_->snapshot());
                try {
                    plan_->complete_read(*stale_window_, ++event_id_,
                                         layer_bundle_bytes(layer));
                } catch (const std::exception & error) {
                    source_service_->mark_retirable(source_ticket);
                    source_service_->retire(source_ticket);
                    drain_unstarted_source_tasks();
                    record_fault(
                        "late_stale_completion_after_reassignment", true,
                        error.what(),
                        {{"before", stale_before},
                         {"after", snapshot_json(plan_->snapshot())},
                         {"stale_window_id", stale_window_->window_id},
                         {"stale_generation", stale_window_->generation},
                         {"current_generation", window.generation},
                         {"h2d_started", false},
                         {"host_ready_published", false}});
                    throw std::runtime_error("injected fatal stale completion");
                }
                throw std::runtime_error("stale completion was unexpectedly accepted");
            }
            if (args_.fault == "wrong_object" && !fault_triggered_) {
                PlanWindowTicket wrong = window;
                wrong.object_id = object_id(layer, (expert + 1) % kExperts);
                const Json wrong_before = snapshot_json(plan_->snapshot());
                try {
                    plan_->complete_read(wrong, ++event_id_,
                                         layer_bundle_bytes(layer));
                } catch (const std::exception & error) {
                    source_service_->mark_retirable(source_ticket);
                    source_service_->retire(source_ticket);
                    drain_unstarted_source_tasks();
                    record_fault(
                        "wrong_object_identity", true, error.what(),
                        {{"before", wrong_before},
                         {"after", snapshot_json(plan_->snapshot())},
                         {"declared_object", object},
                         {"injected_object", wrong.object_id},
                         {"h2d_started", false},
                         {"host_ready_published", false}});
                    throw std::runtime_error("injected fatal wrong-object completion");
                }
                throw std::runtime_error("wrong-object completion was unexpectedly accepted");
            }
        } else {
            read_exact(fd_, layers_[layer].gate.source_offset +
                            expert * layers_[layer].gate.expert_bytes,
                       host_window_pointer(0), layers_[layer].gate.expert_bytes);
            read_exact(fd_, layers_[layer].up.source_offset +
                            expert * layers_[layer].up.expert_bytes,
                       host_window_pointer(0) + gate_bytes_, layers_[layer].up.expert_bytes);
            read_exact(fd_, layers_[layer].down.source_offset +
                            expert * layers_[layer].down.expert_bytes,
                       host_window_pointer(0) + gate_bytes_ + up_bytes_,
                       layers_[layer].down.expert_bytes);
        }
        if (!source_service_ && args_.drop_source_cache) {
            const std::array<std::pair<std::uint64_t, std::uint64_t>, 3> ranges{{
                {layers_[layer].gate.source_offset + expert * layers_[layer].gate.expert_bytes,
                 layers_[layer].gate.expert_bytes},
                {layers_[layer].up.source_offset + expert * layers_[layer].up.expert_bytes,
                 layers_[layer].up.expert_bytes},
                {layers_[layer].down.source_offset + expert * layers_[layer].down.expert_bytes,
                 layers_[layer].down.expert_bytes},
            }};
            for (const auto & [offset, bytes] : ranges) {
                const int status = posix_fadvise(fd_, static_cast<off_t>(offset),
                                                 static_cast<off_t>(bytes),
                                                 POSIX_FADV_DONTNEED);
                require(status == 0, "source cache discard failed: " +
                                         std::string(std::strerror(status)));
                ++source_cache_advice_calls_;
                source_cache_advice_bytes_ += bytes;
            }
        }
        auto end = Clock::now();
        timings_.source_ns += elapsed_ns(begin, end);
        const std::uint64_t materialized_bytes = layer_bundle_bytes(layer);
        materialized_bytes_ += materialized_bytes;
        plan_->complete_read(window, ++event_id_, materialized_bytes);

        const auto copy = plan_->begin_copy(window, kSlotId, 0);
        if (source_service_) {
            source_service_->begin_h2d(source_ticket, materialized_bytes);
        }
        begin = Clock::now();
        ggml_backend_tensor_set(plan_tensor_, host_window_pointer(window_index),
                                0, materialized_bytes);
        ggml_backend_synchronize(backend_);
        end = Clock::now();
        timings_.h2d_ns += elapsed_ns(begin, end);
        plan_->complete_copy(copy, ++event_id_, materialized_bytes);
        if (source_service_) {
            source_service_->complete_h2d(source_ticket, materialized_bytes, event_id_);
        }
        const auto consumer = plan_->acquire(object, 0);
        if (args_.fault == "held_consumer" && !fault_triggered_) {
            expect_nonfatal_rejection(
                "recycle_with_held_consumer",
                [this, &window] { plan_->recycle_window(window); },
                {{"consumer_ticket_id", consumer.ticket_id},
                 {"window_id", window.window_id}});
        }
        if (args_.fault == "shutdown_live_consumer" && !fault_triggered_) {
            expect_nonfatal_rejection(
                "shutdown_with_live_consumer",
                [this] { plan_->shutdown(); },
                {{"consumer_ticket_id", consumer.ticket_id},
                 {"window_id", window.window_id}});
        }

        const char * source = static_cast<const char *>(plan_tensor_->data);
        copy_to_destination(layer, expert, source);

        if (populate_cache) {
            CacheEntry & entry = cache_entries_[layer][cache_slot];
            const bool evicted = entry.valid;
            begin = Clock::now();
            copy_d2d(cache_slot_pointer(layer, cache_slot), source, materialized_bytes);
            synchronize_d2d();
            timings_.cache_fill_ns += elapsed_ns(begin, Clock::now());
            record_cache_population(layer, entry, expert, evicted);
        }

        plan_->complete(consumer, ++event_id_);
        plan_->release_slot(kSlotId);
        plan_->recycle_window(window);
        if (source_service_) {
            source_service_->retire(source_ticket);
        }
        if (args_.fault == "stale_completion" && !stale_window_.has_value()) {
            stale_window_ = window;
        }
        ++materializations_;
    }

    void materialize_synchronous(int layer, int expert, bool populate_cache,
                                 std::size_t cache_slot) {
        read_synchronous_source(layer, expert);
        const std::uint64_t materialized_bytes = layer_bundle_bytes(layer);
        materialized_bytes_ += materialized_bytes;
        auto begin = Clock::now();
        simple_source_.h2d_issued_bytes += materialized_bytes;
        ggml_backend_tensor_set(plan_tensor_, host_window_pointer(0),
                                0, materialized_bytes);
        ggml_backend_synchronize(backend_);
        simple_source_.h2d_completed_bytes += materialized_bytes;
        PhaseSource & phase = current_step_ == 0 ? prefill_source_ : decode_source_;
        phase.h2d_completed_bytes += materialized_bytes;
        timings_.h2d_ns += elapsed_ns(begin, Clock::now());

        const char * source = static_cast<const char *>(plan_tensor_->data);
        copy_to_destination(layer, expert, source);
        if (populate_cache) {
            CacheEntry & entry = cache_entries_[layer][cache_slot];
            const bool evicted = entry.valid;
            begin = Clock::now();
            copy_d2d(cache_slot_pointer(layer, cache_slot), source, materialized_bytes);
            synchronize_d2d();
            timings_.cache_fill_ns += elapsed_ns(begin, Clock::now());
            record_cache_population(layer, entry, expert, evicted);
        }
        ++materializations_;
    }

    const Args & args_;
    llama_model * model_;
    std::array<LayerGeometry, kLayers> layers_{};
    std::uint64_t gate_bytes_ = 0;
    std::uint64_t down_bytes_ = 0;
    int down_q4_layers_ = 0;
    int down_q6_layers_ = 0;
    std::uint64_t up_bytes_ = 0;
    std::uint64_t bundle_bytes_ = 0;
    std::uint64_t source_file_bytes_ = 0;
    std::uint64_t managed_expert_inventory_bytes_ = 0;
    std::uint64_t shared_destination_raw_bytes_ = 0;
    int fd_ = -1;
    void * host_arena_ = nullptr;
    std::size_t host_arena_bytes_ = 0;
    std::size_t host_window_count_ = 0;
    std::uint64_t host_window_stride_ = 0;
    std::uint64_t direct_scratch_offset_ = 0;
    std::uint64_t direct_scratch_bytes_ = 0;
    bool host_windows_registered_ = false;
    std::unique_ptr<SourceService> source_service_;
    SimpleSourceTelemetry simple_source_{};
    SourceTelemetry step_source_begin_{};
    bool step_source_snapshot_ = false;
    PhaseSource prefill_source_{};
    PhaseSource decode_source_{};
    std::unordered_map<int, PendingSource> pending_source_;
    ggml_backend_t backend_ = nullptr;
    ggml_context * plan_ctx_ = nullptr;
    ggml_tensor * plan_tensor_ = nullptr;
    ggml_backend_buffer_t plan_buffer_ = nullptr;
    std::uint64_t plan_buffer_bytes_ = 0;
    std::unique_ptr<Engine> plan_;
    bool cache_enabled_ = false;
    void * cache_data_ = nullptr;
    std::uint64_t cache_bytes_ = 0;
    std::array<std::vector<CacheEntry>, kLayers> cache_entries_{};
    std::vector<std::uint64_t> max_cache_occupancy_;
    bool step_active_ = false;
    bool shutdown_ = false;
    std::uint64_t current_step_ = 0;
    std::uint64_t current_chunk_ = 0;
    std::uint64_t plan_request_sequence_ = 0;
    std::uint64_t event_id_ = 0;
    std::uint64_t materializations_ = 0;
    std::uint64_t materialized_bytes_ = 0;
    std::uint64_t cache_hit_bytes_ = 0;
    std::uint64_t source_cache_advice_calls_ = 0;
    std::uint64_t source_cache_advice_bytes_ = 0;
    std::uint64_t cache_clock_ = 0;
    std::uint64_t cache_generation_ = 0;
    std::uint64_t cache_accesses_ = 0;
    std::uint64_t cache_hits_ = 0;
    std::uint64_t cache_misses_ = 0;
    std::uint64_t prefill_cache_accesses_ = 0;
    std::uint64_t prefill_cache_hits_ = 0;
    std::uint64_t prefill_cache_misses_ = 0;
    std::uint64_t decode_cache_accesses_ = 0;
    std::uint64_t decode_cache_hits_ = 0;
    std::uint64_t decode_cache_misses_ = 0;
    std::uint64_t cache_populations_ = 0;
    std::uint64_t decode_cache_populations_ = 0;
    std::uint64_t cache_evictions_ = 0;
    std::array<std::uint64_t, kLayers> cache_populations_per_layer_{};
    std::array<std::uint64_t, kLayers> decode_cache_populations_per_layer_{};
    std::array<std::uint64_t, kLayers> decode_cache_recycles_per_layer_{};
    std::uint64_t prefill_tail_population_skips_ = 0;
    std::uint64_t cache_stale_rejections_ = 0;
    std::uint64_t cache_premature_reuse_rejections_ = 0;
    std::set<int> layers_seen_;
    std::string callback_error_;
    bool fault_triggered_ = false;
    bool fault_fatal_ = false;
    std::string fault_stage_;
    std::string fault_rejection_;
    Json fault_details_ = Json::object();
    Json fault_terminal_snapshot_ = nullptr;
    Json fault_terminal_source_ = nullptr;
    std::optional<PlanWindowTicket> stale_window_;
    Timings timings_;
    Json route_events_ = Json::array();
    Json pre_shutdown_ = nullptr;
    Json shutdown_snapshot_ = nullptr;
};

std::string format_prompt(const llama_model * model, const std::string & user_prompt,
                          bool disable_thinking) {
    const llama_chat_message message{"user", user_prompt.c_str()};
    const char * chat_template = llama_model_chat_template(model, nullptr);
    require(chat_template != nullptr, "official model has no chat template");
    std::vector<char> buffer(std::max<std::size_t>(4096, user_prompt.size() * 4));
    int32_t length = llama_chat_apply_template(chat_template, &message, 1, true,
                                               buffer.data(), buffer.size());
    require(length >= 0, "official Qwen3-Next chat template is unsupported");
    if (length > static_cast<int32_t>(buffer.size())) {
        buffer.resize(length);
        length = llama_chat_apply_template(chat_template, &message, 1, true,
                                           buffer.data(), buffer.size());
        require(length >= 0 && length <= static_cast<int32_t>(buffer.size()),
                "chat template resize failed");
    }
    std::string formatted(buffer.data(), length);
    if (disable_thinking) {
        require(std::string(chat_template).find(
                    "enable_thinking is defined and enable_thinking is false") !=
                    std::string::npos,
                "official model template has no thinking-disabled branch");
        // llama_chat_apply_template intentionally uses its recognized Qwen C++
        // renderer rather than evaluating the Jinja-only enable_thinking variable.
        // Its output ends at the assistant header, so append the exact false-branch
        // prefix embedded in this GGUF instead of editing a model-generated token.
        const std::string assistant_suffix = "<|im_start|>assistant\n";
        require(formatted.size() >= assistant_suffix.size() &&
                    formatted.compare(formatted.size() - assistant_suffix.size(),
                                      assistant_suffix.size(), assistant_suffix) == 0,
                "official default thinking prompt has an unexpected suffix");
        formatted.append("<think>\n\n</think>\n\n");
    }
    return formatted;
}

std::string prompt_text(const Args & args) {
    if (args.prompt_file.empty()) {
        return args.prompt;
    }
    std::ifstream input(args.prompt_file);
    require(static_cast<bool>(input), "cannot open prompt file: " + args.prompt_file);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & prompt) {
    const int32_t count = llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0,
                                         true, true);
    require(count < 0, "token count query failed");
    std::vector<llama_token> tokens(static_cast<std::size_t>(-count));
    const int32_t actual = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(),
                                          tokens.size(), true, true);
    require(actual == static_cast<int32_t>(tokens.size()), "prompt tokenization failed");
    return tokens;
}

bool resident_router_callback(ggml_tensor * tensor, bool ask, void *) {
    int layer = -1;
    if (sscanf(ggml_get_name(tensor), "ffn_moe_topk-%d", &layer) != 1 ||
        layer < 0 || layer >= kLayers) {
        return false;
    }
    (void)ask;
    return true;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::array<char, 256> local{};
    int32_t count = llama_token_to_piece(vocab, token, local.data(), local.size(), 0, true);
    if (count >= 0) {
        return std::string(local.data(), count);
    }
    std::vector<char> expanded(static_cast<std::size_t>(-count));
    count = llama_token_to_piece(vocab, token, expanded.data(), expanded.size(), 0, true);
    require(count >= 0, "token-to-piece conversion failed");
    return std::string(expanded.data(), count);
}

Json model_memory_json(const llama_model * model) {
    Json result = Json::array();
    for (const auto & [buffer_type, bytes] : model->memory_breakdown()) {
        result.push_back({
            {"buffer_type", ggml_backend_buft_name(buffer_type)},
            {"bytes", bytes},
        });
    }
    return result;
}

Json run_request(const Args & args) {
    const ProcessIo io_begin = process_io();
    const auto initialization_begin = Clock::now();
    ggml_backend_load_all_from_path(args.backend_dir.c_str());
    const bool plan_arm = is_streamed_arm(args);
    if (plan_arm) {
        require(setenv("MSI_QWEN3NEXT_STREAM_EXPERTS", "1", 1) == 0,
                "cannot enable streamed expert binding");
    } else {
        unsetenv("MSI_QWEN3NEXT_STREAM_EXPERTS");
    }
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = args.n_gpu_layers;
    model_params.load_mode = args.direct_model_load ? LLAMA_LOAD_MODE_DIRECT_IO :
                             args.no_mmap ? LLAMA_LOAD_MODE_NONE : LLAMA_LOAD_MODE_MMAP;
    llama_model * model = llama_model_load_from_file(args.model.c_str(), model_params);
    require(model != nullptr, "model load failed");

    std::unique_ptr<Qwen3NextPlanStreamer> streamer;
    std::unique_ptr<Qwen3NextRouteObserver> route_observer;
    if (plan_arm) {
        streamer = std::make_unique<Qwen3NextPlanStreamer>(args, model);
    } else if (args.arm == "resident_trace_control") {
        route_observer = std::make_unique<Qwen3NextRouteObserver>();
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string formatted = format_prompt(model, prompt_text(args), args.disable_thinking);
    std::vector<llama_token> prompt_tokens = tokenize(vocab, formatted);
    if (args.prompt_token_limit > 0 &&
        prompt_tokens.size() > static_cast<std::size_t>(args.prompt_token_limit)) {
        prompt_tokens.resize(static_cast<std::size_t>(args.prompt_token_limit));
    }
    require(prompt_tokens.size() >= 8, "natural request prompt is unexpectedly small");
    require(!is_fault_arm(args) || prompt_tokens.size() == 512,
            "fault canary prompt must resolve to exactly 512 tokens");

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max<std::uint32_t>(256, prompt_tokens.size() + args.n_predict + 8);
    context_params.n_batch = prompt_tokens.size();
    context_params.n_ubatch = std::min<std::size_t>(args.n_ubatch, prompt_tokens.size());
    context_params.no_perf = false;
    if (streamer) {
        context_params.cb_eval = Qwen3NextPlanStreamer::callback;
        context_params.cb_eval_user_data = streamer.get();
    } else if (route_observer) {
        context_params.cb_eval = Qwen3NextRouteObserver::callback;
        context_params.cb_eval_user_data = route_observer.get();
    } else if (args.arm == "resident_observed_control") {
        context_params.cb_eval = resident_router_callback;
    }
    llama_context * context = llama_init_from_model(model, context_params);
    require(context != nullptr, "context initialization failed");
    const Json model_memory = model_memory_json(model);

    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    require(sampler != nullptr, "sampler initialization failed");
    std::vector<llama_logit_bias> eog_biases;
    if (args.fixed_horizon) {
        eog_biases.reserve(8);
        for (llama_token token = 0; token < llama_vocab_n_tokens(vocab); ++token) {
            if (llama_vocab_is_eog(vocab, token)) {
                eog_biases.push_back(
                    {token, -std::numeric_limits<float>::infinity()});
            }
        }
        require(!eog_biases.empty(), "fixed-horizon sampler found no EOG tokens");
        llama_sampler_chain_add(sampler, llama_sampler_init_logit_bias(
            llama_vocab_n_tokens(vocab), static_cast<int32_t>(eog_biases.size()),
            eog_biases.data()));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    const auto initialization_end = Clock::now();

    std::ofstream logits(args.logits, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(logits), "cannot open logits output");
    const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> generated;
    std::vector<std::string> generated_pieces;
    std::vector<std::uint64_t> output_ready_request_elapsed_ns;
    std::string response;
    std::uint64_t prefill_ns = 0;
    std::uint64_t decode_ns = 0;
    std::uint64_t step = 0;
    std::uint64_t logits_rows = 0;
    bool fatal_fault = false;
    const auto request_begin = Clock::now();
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    while (generated.size() < static_cast<std::size_t>(args.n_predict)) {
        if (streamer) {
            streamer->begin_step(step + 1);
        }
        if (route_observer) {
            route_observer->begin_step(step);
        }
        const auto decode_begin = Clock::now();
        const int status = llama_decode(context, batch);
        const auto decode_end = Clock::now();
        if (streamer && streamer->fatal_fault_triggered()) {
            fatal_fault = true;
            break;
        }
        if (status != 0) {
            if (streamer) {
                streamer->abort_step();
            }
            throw std::runtime_error("stock llama_decode failed with status " + std::to_string(status));
        }
        if (streamer) {
            streamer->finish_step();
        }
        if (route_observer) {
            route_observer->finish_step();
        }
        if (step == 0) {
            prefill_ns += elapsed_ns(decode_begin, decode_end);
        } else {
            decode_ns += elapsed_ns(decode_begin, decode_end);
        }
        float * current_logits = llama_get_logits_ith(context, -1);
        require(current_logits != nullptr, "stock runtime returned no logits");
        logits.write(reinterpret_cast<const char *>(current_logits),
                     static_cast<std::streamsize>(vocabulary_size * sizeof(float)));
        require(static_cast<bool>(logits), "logits write failed");
        ++logits_rows;
        const llama_token token = llama_sampler_sample(sampler, context, -1);
        if (!args.fixed_horizon && llama_vocab_is_eog(vocab, token)) {
            break;
        }
        require(!args.fixed_horizon || !llama_vocab_is_eog(vocab, token),
                "fixed-horizon EOG masking failed");
        generated.push_back(token);
        generated_pieces.push_back(token_piece(vocab, token));
        response += generated_pieces.back();
        output_ready_request_elapsed_ns.push_back(elapsed_ns(request_begin, Clock::now()));
        batch = llama_batch_get_one(&generated.back(), 1);
        ++step;
    }
    logits.close();
    require(file_size(args.logits) ==
                logits_rows * static_cast<std::uint64_t>(vocabulary_size) * sizeof(float),
            "logits output size mismatch");

    llama_memory_clear(llama_get_memory(context), true);
    if (streamer && !fatal_fault) {
        streamer->shutdown();
    }
    const auto shutdown_begin = Clock::now();
    Json plan_report = streamer ? streamer->report() : Json(nullptr);
    Json injected_fault_report = streamer ? streamer->fault_report() : Json(nullptr);
    Json route_trace = route_observer ? route_observer->report() : Json(nullptr);
    llama_sampler_free(sampler);
    llama_free(context);
    streamer.reset();
    llama_model_free(model);
    const auto shutdown_end = Clock::now();
    const ProcessIo io_end = process_io();

    return {
        {"schema_version", 1},
        {"artifact", "qwen3_next_plan_complete_request_arm"},
        {"arm", args.arm},
        {"consumer", {
            {"base_runtime_commit", "8ef78e644f559db4e8716b59bf76b8e11619337d"},
            {"runtime_adapter_commit", "7206605099424f7ad7ca78c2783a60ec60f810f4"},
            {"graph", plan_arm ? "stock_qwen3_next_llama_decode" : "stock_model_llama_decode"},
            {"router", "stock_ffn_moe_topk"},
            {"expert_consumer", plan_arm ? "stock_ggml_mul_mat_id_q4_k_q6_k" :
                "stock_runtime_model_consumer"},
            {"reduction", plan_arm ? "stock_qwen3_next_moe_reduction" :
                "stock_runtime_model_reduction"},
            {"scheduler_boundary", args.arm == "resident_control" ?
                "ordinary_unsplit" : "pause_after_each_ffn_moe_topk"},
        }},
        {"model_memory", model_memory},
        {"workload", {
            {"prompt", args.prompt},
            {"prompt_file", args.prompt_file},
            {"prompt_token_limit", args.prompt_token_limit},
            {"prompt_formatter", "official_model_one_user_no_tools_chat_template"},
            {"thinking_mode", args.disable_thinking ? "official_disabled" :
                                                       "template_default"},
            {"sampling", args.fixed_horizon ? "greedy_eog_masked_fixed_horizon" :
                                                "greedy_natural_eog"},
            {"eog_tokens_masked", eog_biases.size()},
            {"formatted_prompt_bytes", formatted.size()},
            {"prompt_tokens", prompt_tokens.size()},
            {"n_predict_limit", args.n_predict},
            {"n_gpu_layers", args.n_gpu_layers},
            {"n_ubatch", args.n_ubatch},
            {"use_mmap", !args.no_mmap && !args.direct_model_load},
            {"model_load_mode", args.direct_model_load ? "direct_io" :
                                args.no_mmap ? "none" : "mmap"},
            {"generated_tokens", generated},
            {"generated_pieces", generated_pieces},
            {"output_ready_request_elapsed_ns", output_ready_request_elapsed_ns},
            {"generated_token_count", generated.size()},
            {"response", response},
            {"logits_path", args.logits},
            {"logits_bytes", file_size(args.logits)},
            {"vocabulary_size", vocabulary_size},
        }},
        {"timing_ns", {
            {"initialization", elapsed_ns(initialization_begin, initialization_end)},
            {"prefill", prefill_ns},
            {"decode", decode_ns},
            {"reset_and_shutdown", elapsed_ns(shutdown_begin, shutdown_end)},
        }},
        {"plan", plan_report},
        {"fault", injected_fault_report},
        {"fault_acceptance", {
            {"fatal_fault", fatal_fault},
            {"accepted_tokens_after_fault", fatal_fault ? generated.size() : 0},
            {"accepted_logits_after_fault", fatal_fault ? logits_rows : 0},
        }},
        {"route_trace", route_trace},
        {"process_io", process_io_delta(io_begin, io_end)},
    };
}

void write_json(const std::string & path, const Json & value) {
    std::ofstream output(path, std::ios::trunc);
    require(static_cast<bool>(output), "cannot open JSON output: " + path);
    output << value.dump(2) << '\n';
    require(static_cast<bool>(output), "cannot write JSON output: " + path);
}

Json preflight_geometry(const std::string & model_path) {
    ggml_context * metadata_ctx = nullptr;
    gguf_init_params params{true, &metadata_ctx};
    gguf_context * gguf = gguf_init_from_file(model_path.c_str(), params);
    require(gguf != nullptr && metadata_ctx != nullptr, "cannot open GGUF metadata");
    require(gguf_get_n_tensors(gguf) == 807, "Qwen3-Next GGUF tensor count changed");
    std::uint64_t inventory_bytes = 0;
    int q4_down_layers = 0;
    int q6_down_layers = 0;
    try {
        for (int layer = 0; layer < kLayers; ++layer) {
            for (const char * suffix : {
                    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
                const std::string name = tensor_name(layer, suffix);
                const int64_t id = gguf_find_tensor(gguf, name.c_str());
                require(id >= 0, "GGUF tensor is missing: " + name);
                const std::uint64_t bytes = gguf_get_tensor_size(gguf, id);
                require(bytes % kExperts == 0, "expert tensor does not split evenly: " + name);
                const ggml_type type = gguf_get_tensor_type(gguf, id);
                const std::uint64_t quantization_block_bytes =
                    type == GGML_TYPE_Q4_K ? 144ULL :
                    type == GGML_TYPE_Q6_K ? 210ULL : 0ULL;
                require(quantization_block_bytes > 0 &&
                            (bytes / kExperts) % quantization_block_bytes == 0,
                        "expert tensor native quantization block changed: " + name);
                if (strcmp(suffix, "ffn_down_exps.weight") == 0) {
                    if (type == GGML_TYPE_Q4_K) {
                        ++q4_down_layers;
                        require(bytes / kExperts == 589824, "Q4_K down expert shape changed");
                    } else if (type == GGML_TYPE_Q6_K) {
                        ++q6_down_layers;
                        require(bytes / kExperts == 860160, "Q6_K down expert shape changed");
                    } else {
                        throw std::runtime_error("unsupported down expert type: " +
                                                 std::string(ggml_type_name(type)));
                    }
                } else {
                    require(type == GGML_TYPE_Q4_K && bytes / kExperts == 589824,
                            "Qwen3-Next gate/up expert type or shape changed");
                }
                const std::uint64_t end = gguf_get_data_offset(gguf) +
                                          gguf_get_tensor_offset(gguf, id) + bytes;
                require(end <= file_size(model_path), "expert tensor exceeds source file: " + name);
                inventory_bytes += bytes;
            }
        }
    } catch (...) {
        gguf_free(gguf);
        ggml_free(metadata_ctx);
        throw;
    }
    gguf_free(gguf);
    ggml_free(metadata_ctx);
    require(q4_down_layers == 24 && q6_down_layers == 24,
            "Qwen3-Next mixed down-expert quantization pattern changed");
    require(inventory_bytes == 46808432640ULL,
            "Qwen3-Next managed expert inventory changed");
    return {
        {"tensor_count", 807},
        {"layers", kLayers},
        {"experts", kExperts},
        {"experts_used", kExpertsUsed},
        {"gate_up_type", "Q4_K"},
        {"q4_k_quantization_block_bytes", 144},
        {"q6_k_quantization_block_bytes", 210},
        {"down_q4_layers", q4_down_layers},
        {"down_q6_layers", q6_down_layers},
        {"managed_expert_inventory_bytes", inventory_bytes},
        {"maximum_expert_bundle_bytes", 2039808},
        {"execution_time_conversion", false},
    };
}

void preflight(const Args & args) {
    const bool plan_arm = is_streamed_arm(args);
    if (plan_arm) {
        require(file_size(args.model) == kOfficialModelBytes,
                "official Qwen3-Next model byte size mismatch");
    } else {
        require(file_size(args.model) > 0, "resident model is missing or empty");
    }
    require(file_size(args.backend_dir + "/libllama.so") > 0,
            "private libllama is missing");
    require(file_size(args.backend_dir + "/libggml-cuda.so.0.20.2") > 0,
            "private CUDA backend is missing");
    write_json(args.output, {
        {"schema_version", 1},
        {"artifact", "qwen3_next_plan_complete_request_preflight"},
        {"status", "PASS"},
        {"arm", args.arm},
        {"fault", args.fault},
        {"model_bytes", file_size(args.model)},
        {"geometry", plan_arm ? preflight_geometry(args.model) : Json(nullptr)},
        {"layers", plan_arm ? Json(kLayers) : Json(nullptr)},
        {"experts", plan_arm ? Json(kExperts) : Json(nullptr)},
        {"experts_used", plan_arm ? Json(kExpertsUsed) : Json(nullptr)},
        {"n_predict", args.n_predict},
        {"cache_capacity", args.cache_capacity},
        {"n_gpu_layers", args.n_gpu_layers},
        {"n_ubatch", args.n_ubatch},
        {"drop_source_cache", args.drop_source_cache},
        {"bounded_source", args.bounded_source},
        {"source_direct", args.source_direct},
        {"source_in_flight", args.source_in_flight},
        {"use_mmap", !args.no_mmap && !args.direct_model_load},
        {"model_load_mode", args.direct_model_load ? "direct_io" :
                            args.no_mmap ? "none" : "mmap"},
        {"thinking_mode", args.disable_thinking ? "official_disabled" :
                                                   "template_default"},
        {"sampling", args.fixed_horizon ? "greedy_eog_masked_fixed_horizon" :
                                            "greedy_natural_eog"},
        {"prompt", args.prompt},
        {"prompt_file", args.prompt_file},
        {"prompt_token_limit", args.prompt_token_limit},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.preflight) {
            preflight(args);
        } else {
            write_json(args.output, run_request(args));
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen3_next_plan_runtime_adapter: " << error.what() << '\n';
        return 1;
    }
}
