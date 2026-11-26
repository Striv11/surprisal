# surprisal

> A brain-inspired sequence learner that predicts what comes next, learns online from its own prediction errors, and lets you watch its *surprise*. Rust, compiled to WebAssembly.

**Live demo:** https://striv11.github.io/surprisal/ (built and deployed by `.github/workflows/pages.yml` on every push to main; requires Pages source set to "GitHub Actions").

![CI](https://github.com/Striv11/surprisal/actions/workflows/ci.yml/badge.svg)

## Problem
Most sequence models are trained offline with backpropagation through time and are opaque. Predictive coding offers another account: a system continuously predicts its input and learns only from the mismatch. People who want to *see* this idea in action (students, curious musicians, cognitive-science fans) have few interactive, honest implementations. surprisal lets you play or type a pattern, watch the model's expectation for every next symbol, and see surprise spike when you break the pattern.

## Motivation
- Implement a predictive-coding network from scratch, with no ML framework and no crates, so every equation is readable.
- Learning uses local rules only: weight change = prediction error x presynaptic activity. No backprop through time.
- Rust core compiled to WASM with a hand-written C ABI (no wasm-bindgen), so the same deterministic code runs in unit tests and in the browser. Everything runs on-device; nothing is uploaded.

## Architecture
```
 symbol o_t ──►┌──────────────────────────────────────────────┐
               │ prior      mu_t = tanh(A s_{t-1} + B o_{t-1}) │
               │ prediction p_t  = softmax(C mu_t)  (before o_t)│
               │ surprise   -ln p_t[o_t]                        │
               │ inference  s_t ← s_t + η(Cᵀe_o − λ(s_t − mu_t))│ × K steps
               │ learning   C += lr·e_o·s_tᵀ ; A,B += lr·λ·e_s·(input)ᵀ │
               └──────────────────────────────────────────────┘
  src/rng.rs    deterministic xorshift PRNG
  src/model.rs  the network (Config, Model, StepReport)
  src/main.rs   terminal demo
  src/wasm.rs   C-ABI exports (sp_create/sp_create2/observe/predict/generate/reset)
  web/src/engine.ts   typed wrapper over the C ABI
  web/src/pattern.ts  note/pattern parsing
  web/src/timeline.ts anomaly detection + canvas surprise timeline
  web/src/main.ts     UI (piano, text input, expectation bars, timeline, dream, history)
```
Invariants: predictions are a probability distribution; the prediction for `o_t` is made before `o_t` influences state; given a seed, training is bit-for-bit reproducible.

## Setup
Prerequisites: a stable Rust toolchain.
```
bash check.sh        # fmt, clippy, tests
cargo run --release  # terminal demo
cargo build --release --lib --target wasm32-unknown-unknown  # browser module
node tests-js/parity.mjs  # wasm == native check
cd web && npm ci && npm run dev    # web app (builds the wasm, then Vite)
cd web && npm test && npm run build  # unit tests, static dist/
```

## Demo
Terminal demo (works today):
```
cargo run --release
epoch  mean surprise (nats)  next-note accuracy
    0                 1.453                 59%
   20                 0.196                100%
  120                 0.046                100%
dreamed continuation: C E G E C E G B C E G E C E G B
```
Browser demo: piano keys, text patterns, expectation bars, a canvas surprise timeline with anomalies marked red (surprise > 1.5 nats and > 3x the earlier median), and a "dream" button that samples a continuation without learning.

## Design decisions & trade-offs
- Optional second level (`Config::top_hidden`): a slow leaky layer `z` integrates level-1 states and sends a top-down term `D z` into the level-1 prior; `D` learns with the same local rule (state error x presynaptic `z`). The recurrence and level-1->2 weights are fixed random (reservoir-style), a deliberate simplification: only top-down weights are learned. On a period-16 tune whose halves share all local contexts, mean surprise drops ~3.5x versus one level.
- Cross-entropy sensory error (softmax readout) instead of Gaussian error: gives a clean probability per symbol and surprise in nats.
- f32 throughout for small WASM size and speed; determinism comes from the custom PRNG and a fixed operation order.

## Roadmap
- [x] Day 1: core model, PRNG, tests, terminal demo
- [x] WASM C-ABI build (28 KB) and Node parity test
- [x] Web app shell: piano + text input, expectation bars, history
- [x] Surprise timeline canvas, dream mode, Pages workflow
- [x] Hierarchical (two-level) model with UI toggle
- [ ] Playwright e2e, benchmarks, Pages deploy
- [ ] Save/load model, presets, polish

## License
MIT. See [LICENSE](LICENSE).
