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

## 2026-11-27 (Day 6)
- Added `examples/bench.rs`: native throughput benchmark (60k symbols, warm-up) for one- and two-level models; exits non-zero below 20k symbols/s (override via SURPRISAL_BENCH_BUDGET). `check.sh` runs it in release, so CI enforces the budget. Measured ~300k/s single-level and ~260k/s two-level, so the budget has >10x headroom and only catches gross regressions.
- Web: theme toggle (auto/light/dark, persisted, canvas redraws with theme colours) and keyboard shortcuts (1-7, F, T, D, R, M). Logic lives in DOM-free `keys.ts`/`theme.ts` with Node tests (modifiers, typing in fields and focused buttons are ignored). Small-screen CSS tweaks for the form and touch targets.
- Not done: Playwright e2e (no browser here and installing one is heavy); the PLAN row is marked partial. Theme/shortcut behaviour and mobile layout are not visually verified in a real browser.
- Next: Day 7 save/load model JSON with an exact round-trip test, presets, README screenshots, error states, v0.1.0 notes; then Playwright e2e if a browser is available.

## 2026-11-28 (Day 7)
- Added `src/persist.rs`: `Model::to_json` / `from_json` with no dependencies (a small JSON reader). Every f32 is stored as its bit pattern, so the round trip is exact. The loader checks the format tag, version, dimension limits, array lengths and `prev`, and returns `Err` instead of panicking; nesting depth is capped.
- C ABI: `sp_save`, `sp_load`, `sp_dims`. `Engine.save()` / `restore()` in TS; UI has Save model (download) and Load model (file picker) with a visible error when the file is invalid or has the wrong vocabulary size. Loading restores the two-level checkbox.
- Presets: added Lullaby and a Random button (`randomPattern`, never repeats a note back to back).
- Tests: Rust round trip for one- and two-level models (re-serialised JSON identical, then 5 further observations and generation match bit-for-bit), 12 malformed-file cases, ABI round trip; Node test for wasm save/restore and rejecting garbage; preset/random pattern tests.
- Learned: the wasm grew from ~31 KB to ~71 KB, mostly from f64 parsing/number formatting pulled in by the JSON code. Still under the 100 KB budget; a bit-pattern-only parser would shrink it if needed.
- Not verified: Save/Load buttons in a real browser (no browser here; web typecheck was skipped because `web/node_modules` is absent, tests ran with node directly).
- Next: README screenshots and v0.1.0 notes once a browser is available; Playwright e2e from Day 6.
