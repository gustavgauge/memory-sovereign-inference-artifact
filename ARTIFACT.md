# Artifact contract

## Evidence boundary

Every packaged object has one of four roles:

1. a compact claim record under `evidence/`;
2. a compact replay input under `evidence/replay/`;
3. a source snapshot under `implementation/frozen/`; or
4. an upstream runtime identity plus an exact patch under `runtime/`.

The claim ledger records which objects support each statement in the paper.
Source reconstruction and execution reproduction are distinct: a portable
build may use a newer compiler or equivalent dependency resolution, but it is
not the evidence-producing executable unless the recorded binary and library
identities also match.

`implementation/source-manifest.json` enumerates and hashes the curated
claim-bearing source closure. Unrelated experiment configurations, project
governance files, and non-claim-bearing development tools are not included.

Recorded contracts and source snapshots retain experiment-time path fields and
implementation identifiers as provenance. They do not prescribe an external
filesystem layout. `workloads/execution-reconstruction.json` identifies the
source, contract, model, workload, and environment bindings for reconstructing
the principal execution paths.

## Commands

`./artifact verify` checks SHA-256 identities, JSON pointers or Markdown
locators, and machine-readable expected assertions. `./artifact replay`
independently recomputes the reported arithmetic using Python's standard
library. Neither command executes a model.

The content manifest is generated and checked with:

```bash
./artifact manifest --write
./artifact manifest --check
```

## Recorded execution identities

| Role | Project commit | Runtime adapter commit | Binary SHA-256 prefix |
|---|---|---|---|
| C1 Qwen3-Next 32K+64 | `70bc7203d92b74e48aa01dbd5700f24ef06ff301` | `7206605099424f7ad7ca78c2783a60ec60f810f4` | `4c68fc99` |
| C2 Qwen3-Next paired D/F | `e495ac509c48f04831ca777d0c9f8f5eafd209a6` | `7206605099424f7ad7ca78c2783a60ec60f810f4` | `7bd52530` |
| C3 Qwen3-Next faults | `4637a2aa099ed355165ad4df2ad5d7eb7974daa1` | `7206605099424f7ad7ca78c2783a60ec60f810f4` | `5083ccd5` |
| C3 Gemma faults | `8902d3c359989f618bb8b5033808b8b00aab346e` | `05f332a0b031aa18041622e26f37e6a7c1fe82f0` | `a01e…` |

The C1 exactness certificate and C3 fault population use different executable
identities. The artifact preserves that boundary and does not attribute the C3
fault transitions to the C1 executable.

## Hardware and model boundary

The principal Qwen3-Next executions used an RTX 3090 with a declared 23 GiB
whole-board budget on a physical 24 GiB board and an 11 GiB hard cgroup-charge
limit over the complete target process hierarchy. This boundary includes
charges that cgroup v2 attributes to the hierarchy; it is not a claim that
Linux attributes every causal host byte. GPU use was observed for the entire
board, not hard-partitioned. Model files are not redistributed;
`environment/models.json` records pinned repositories, revisions, filenames,
byte lengths, and immutable digests. Provider licenses and access terms still
govern acquisition and use.

## Packaged evidence boundary

Compact evidence and all paper-critical replay operands are stored directly in
this repository. Verification checks packaged record identities and enumerated
assertions; replay recomputes arithmetic and categorical summaries from compact
publication operands. Bulk logs, model weights, and complete output binaries
are not redistributed. Their digests preserve the recorded object identities,
but the artifact does not rederive every certificate from raw execution output.
