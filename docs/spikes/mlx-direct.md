# Direct official MLX result (#6)

**Status:** direct official-MLX path technically feasible; third first-class
inference candidate.

**Direction reaffirmed:** 2026-07-17.

**Runnable artifact:** [`spikes/mlx-direct`](../../spikes/mlx-direct/).

## Decision

Cuttledoc will evaluate MLX through a small repository-owned C++ adapter over
the official MLX C++ core. This is a strategic candidate for local Apple
Silicon inference beyond the initial STT selection; it is not rejected because
the Rust boundary is native code.

`mlx-c` is not the planned product integration path. It remains an optional,
pinned reference/control only when it answers a named comparison question.
`mlx-rs`, OminiX-MLX, and `mlx-node` remain implementation references rather
than dependencies: an additional wrapper would not reduce the ownership,
packaging, or upgrade responsibility that Cuttledoc needs to retain.

## Pinned implementation

| Item | Value |
| --- | --- |
| Upstream | [`ml-explore/mlx`](https://github.com/ml-explore/mlx) |
| Tag | `v0.32.0` |
| Commit | `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247` |
| Boundary | Rust-owned primitive buffers -> owned C ABI -> C++ MLX adapter |
| Excluded | MLX arrays, streams, operators, model objects, and general runtime APIs |

The initial adapter performs a dense `[1, 576] x [576, 4]` audio projection
and `tanh` activation. It copies both caller-owned input buffers before MLX
captures its lazy graph, explicitly selects CPU or GPU, evaluates the graph,
and copies four output scores back to Rust. It deliberately does not claim to
be a transcriber or to define the eventual model ABI.

## Build findings

On the macOS 26.5.2 Apple Silicon evidence host, MLX `v0.32.0` configured
against Xcode 26.6 and the macOS 26.5 SDK. The initial attempt exposed a local
toolchain prerequisite: the Xcode Metal Toolchain was absent, so `xcrun metal`
could not compile MLX kernels. After Xcode's documented first-launch
initialization and Metal Toolchain installation, MLX configured with Metal
version 400 and built its static library plus metallib successfully. This is a
host setup requirement, not an MLX architecture blocker.

The checked-in runner verifies the immutable upstream commit, creates a fresh
temporary build, builds the direct adapter, colocates `mlx.metallib`, compiles
the Rust caller, and runs CPU and GPU projections.

## Execution result

The direct Rust -> C ABI -> C++ -> official MLX path completed on the evidence
host with the same synthetic input and weights on both devices:

```text
device=cpu scores=[0.10830319, 0.07250145, 0.03650766, 0.000416309]
device=gpu scores=[0.10830313, 0.07250141, 0.036507454, 0.0004163026]
```

The small CPU/GPU differences are ordinary `float32` rounding; this is a
successful execution and output-copy check, not an output-quality benchmark.
It proves that the narrow adapter can select an MLX device, materialize the
lazy graph, run a dense audio-model block, release the local graph, and return
Rust-owned values without exposing an MLX type.

The static `libmlx.a` was 31 MiB in this configuration. Its Metal kernel
library was a separate 162,449,848-byte `mlx.metallib` (about 155 MiB); the
evidence shim was 18.9 MiB and its Rust caller 521 KiB before release-artifact
optimization. MLX fails at runtime if the metallib is absent, so the runner now
colocates it explicitly. A production adapter still needs kernel selection and
artifact-size work; a successful static link is not a complete distribution
result.

## Remaining evidence

1. Replace the synthetic dense projection with a real, licensed model block
   and record its model format/loading path.
2. Measure repeated load/run/destroy, memory, cold/warm and first-result
   latency, output quality, binary/metallib packaging, and cancellation.
3. Verify upgrade effort across two official MLX releases.
4. Use `mlx-c` only if it resolves a specific unresolved interface/lifecycle
   question; it is not a mandatory product-path comparison.
5. Make the adopt/continue/defer decision separately for STT, text generation,
   and future TTS rather than exposing a generic MLX runtime through the public
   API.

See [the Apple runtime evaluation](../apple-runtime-evaluation.md) and
[the dependency policy](../dependency-policy.md) for the cross-candidate
decision rule.
