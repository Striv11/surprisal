# Work log

Newest entries at the bottom. Each entry records what actually changed, what was learned, and what comes next.

## 2026-11-23 (Day 1)
- Chose "predictive-coding sequence learner" from the Cognition backlog section (Rust, differing from recent projects); named it surprisal.
- Wrote the dependency-free Rust core: xorshift PRNG and a temporal predictive-coding network (prior, softmax prediction, iterative inference, local weight updates, generation).
- Tests: surprise falls >4x on a melody, spikes on an out-of-pattern note, deterministic training, generation reproduces a cycle, vocab bounds.
- Terminal demo shows mean surprise 1.45 -> 0.05 nats over 120 epochs.
- Learned: with a tanh prior and a cross-entropy readout the plain local rules converge quickly without tuning.
- Next: C-ABI WASM exports and a Node parity test (Day 2). Pages deploy and UI come later in the arc.

## 2026-11-24 (Day 2)
- Added `src/wasm.rs`: hand-written C ABI (create, free, observe, predict, generate, reset, alloc/dealloc) with input validation; compiled on all targets so it is unit-tested natively.
- Crate now builds as rlib + cdylib; release profile uses panic=abort, strip, 1 codegen unit. The wasm is 28 KB (budget 100 KB).
- `tests-js/parity.mjs` loads the .wasm in Node and matches native surprise values (1e-5), predictions sum to 1, and generation is identical. `check.sh` runs it when node and the wasm32 target exist.
- Learned: with f32 and a fixed operation order, wasm and native agree closely enough that generation sequences match exactly.
- Next: Day 3 web shell (Vite + TypeScript, loader, piano and text input, expectation bars).

## 2026-11-24 (Day 3)
- Added `web/` (Vite + TypeScript, no runtime deps): `Engine` wrapper over the C ABI, pattern parser (note names or degrees 1-7, unknown characters reported rather than dropped), page with clickable piano, text pattern input, presets, "Train x20", expectation bars and a history strip where surprising notes turn red.
- `npm run build` compiles the wasm, typechecks and emits a static `dist/` (base "./" so it works under a Pages sub-path). The wasm is copied to `public/` and gitignored.
- Tests (`node --test` with native type stripping): parser cases and an engine test that loads the real wasm: learns a melody, predicts the next note, spikes >5x on a wrong note, bad input errors, dream follows learned transitions. `check.sh` now runs typecheck, tests and build when `web/node_modules` exists.
- Learned: exact dream equality is brittle because generation samples; asserting the share of learned transitions is the honest check. Node strip-types forbids TS parameter properties.
- Not verified: visual rendering in a real browser (no browser here); Playwright e2e is planned for Day 6.
- Next: Day 4 canvas surprise timeline, dream button, pages.yml deploy.
