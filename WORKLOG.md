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

## 2026-11-25 (Day 4)
- Added `web/src/timeline.ts`: `findAnomalies` (surprise > 1.5 nats and > 3x the median of earlier values) and a HiDPI canvas line chart with red anomaly markers; the history strip uses the same rule so both views agree. Replaces the fixed threshold from Day 3.
- Added a Dream section (16 sampled notes from the current context, no learning, new seed per click) and a more helpful empty state.
- Added `.github/workflows/pages.yml` (tests, build, upload `web/dist`, deploy-pages); README links the expected URL.
- Fixed: `npm run build:wasm` failed on a fresh checkout because `web/public/` is gitignored and absent; it now `mkdir -p`s it (this would have broken the Pages build).
- Tests: 4 anomaly cases (spike after learned stretch, sub-floor jitter ignored, unlearned high stream flagged only at the first symbol, empty).
- Not verified: canvas rendering in a browser and the Pages deploy itself (no browser or GitHub access here); Pages source must be set to "GitHub Actions" in repo settings.
- Next: Day 5 two-level hierarchy (level-2 predicts level-1 state) with a test on period-16 structure.

## 2026-11-26 (Day 5)
- Added an optional second level to `Model` (`top_hidden`, `top_leak`, `top_lr`): a leaky level-2 state integrates inferred level-1 states and feeds a learned top-down term into the level-1 prior. Default (`top_hidden = 0`) is bit-identical to before, so Day 2 parity still holds.
- New C export `sp_create2(vocab, hidden, top, seed)` (`sp_create` delegates); `Engine.load(..., top)`, `withTop`, and a "Two-level model" checkbox in the UI (toggling starts a fresh model).
- Tests: Rust test on a period-16 tune (halves share every local context) over 3 seeds: two-level mean surprise < 0.6x single-level; determinism/reset test; wasm-side test of the same in Node; ABI size validation.
- Learned: with the default top-down learning rate (1x) the gain was only ~5%; a 10x multiplier on the top-down weights gave ~3.6x lower surprise (0.165 -> 0.045), since they start at zero and learn slowly. Slow leaks (0.1) were worse than 0.4 for this tune. Only top-down weights learn; recurrence is a fixed random reservoir.
- Not verified: the toggle in a real browser; behaviour on real melodies beyond the synthetic tune.
- Next: Day 6 Playwright e2e, theme/keyboard/mobile, and a native throughput benchmark with a CI budget.
