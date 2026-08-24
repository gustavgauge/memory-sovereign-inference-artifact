# Memory-Sovereign Inference

This repository contains the paper and research artifact for *Memory-Sovereign
Inference: Output-Exact Execution Beyond Full Residency*. The paper presents an
execution-certification methodology and reference realization for
storage-backed inference. Its records distinguish which weights exist and are
demanded, how each memory tier is bounded, what exactness comparison was
performed, and which buffer-reuse transitions were tested.

- [Read the paper](paper/preprint.pdf)
- [Inspect the claim ledger](claims/claim-ledger.json)
- [Read the artifact contract](ARTIFACT.md)

## Principal results

Three results form the principal evidence package:

- At one pinned Qwen3-Next Q4_K_M workload and system identity, authoritative
  router outputs show that the workload consumes at least 43.59375 GiB of
  unique managed expert weights after duplicate and overlap checks. This lower
  bound exceeds both the declared 34 GiB host–GPU residency envelope and the
  combined 11 GiB hard-host/24 GiB physical-device envelope. The storage-backed
  execution remains within its stated resource contract and matches a
  prespecified zero-cache oracle across all 64 complete logit rows, token IDs,
  response bytes, 3,408 recorded route events, and the recorded consumer and
  destination identities. See [C1](claims/records/C1.json).
- In a separate matched source experiment, a buffered implementation completes
  exactly but reaches the 11 GiB host limit and records 33,481 `memory.max`
  events. Both direct-source configurations complete exactly with positive
  margin and no limit events. Across six counterbalanced pairs, the complete
  eight-window asynchronous configuration finishes in 32.3% of the blocking
  one-window direct reader's wall time at identical physical source bytes per
  output. See [C2](claims/records/C2.json).
- In separate later fault-instrumented executables, all fourteen prespecified
  Qwen3-Next and Gemma control/fault cells satisfy their validators. These are
  named transition results, not a reliability rate or a fault claim about the
  earlier C1 executable. See [C3](claims/records/C3.json).

These are fixed-identity capability results. They do not establish population
effects across prompts, models, runtimes, or hardware, and the paper does not
claim a new offload primitive. Claims C4–C12 preserve the supporting temporal,
Qwen-specific, Gemma, parity, and negative-result evidence.

## Audit and replay

Audit and replay require Python 3.10 or newer and do not execute a model or
download model weights:

```bash
./artifact verify --all
./artifact replay --all
./artifact manifest --check
```

`verify` checks evidence digests, cited locations, execution identities, and
expected assertions. `replay` independently recomputes the reported arithmetic
and categorical outcomes from the packaged observations. To inspect one claim:

```bash
./artifact verify --claim C1
./artifact replay --claim C2
```

The [claim ledger](claims/claim-ledger.json) maps every paper claim to its
evidence record, content digest, expected assertions, and stable commands.
Within `claims/expected/*.json`, `json_pointer` is an RFC 6901 pointer into the
claim's primary evidence record unless an assertion supplies an explicit
repository-relative `source`.

## Execution reconstruction

The repository includes the recorded source snapshots, runtime patches,
experiment contracts, model identities, and workload identities for the
principal C1–C3 paths. Reconstruction requires acquiring the models listed in
[environment/models.json](environment/models.json), compatible NVIDIA
hardware, the recorded resource controls, and locally built runtime
executables. Each reconstructed run has its own execution identity.

Start with [workloads/execution-reconstruction.json](workloads/execution-reconstruction.json)
for the required bindings and [runtime/reconstruct.sh](runtime/reconstruct.sh)
for the pinned upstream runtime checkouts and patches. Model weights are not
redistributed.

## Repository layout

- [`paper/`](paper/) contains the manuscript, bibliography, and compiled PDF.
- [`claims/`](claims/) contains the paper-to-artifact ledger, expected
  assertions, and compact claim records.
- [`evidence/`](evidence/) contains the paper-critical replay operands.
- [`implementation/`](implementation/) contains the recorded source and
  contract snapshots.
- [`runtime/`](runtime/) contains upstream runtime identities, patches, and the
  reconstruction helper.
- [`environment/`](environment/) records hardware, model, and toolchain
  identities.
- [`workloads/`](workloads/) specifies the principal execution bindings.

## Evidence boundary

The repository includes compact observations, digests, validators, claim
records, and source identities. Verification checks record identities and
enumerated assertions; replay recomputes arithmetic and categorical summaries
from the packaged operands. Model weights, bulk logs, and complete logit
binaries are not redistributed. Their digests preserve the recorded object
identities, but the artifact does not rederive every certificate from raw
execution output.

The C1 full-output comparison and C3 fault-transition population use different
executables. Their evidence remains separate throughout the paper and claim
ledger. [ARTIFACT.md](ARTIFACT.md) defines the complete evidence boundary,
resource authorities, and recorded execution identities.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The immutable
artifact release is [`v0.1.1`](https://github.com/gustavgauge/memory-sovereign-inference-artifact/releases/tag/v0.1.1).

## Licensing

Original software and validators are licensed under Apache-2.0. The paper,
documentation, and evidence metadata are licensed under CC BY 4.0. Third-party
material and model files retain their own terms; see
[`LICENSES/README.md`](LICENSES/README.md).

## Author

Lukas Stepanek, Independent Researcher, <luki.step@proton.me>,
[krooksn.dev](https://krooksn.dev)
