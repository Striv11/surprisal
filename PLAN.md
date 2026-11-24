# Plan

Seven-day arc. Every milestone has acceptance criteria that can be checked by a test or a command.

| Day | Milestone | Acceptance criteria | Status |
|-----|-----------|---------------------|--------|
| 1 | Core model + terminal demo | `bash check.sh` passes; tests show surprise falls >4x on a repeating melody, spikes >5x on a wrong note, training is deterministic; `cargo run --release` prints a falling surprise table | [x] |
| 2 | WASM build | `src/wasm.rs` exports create/observe/predict/surprise/generate/reset via C ABI; `wasm32-unknown-unknown` release build < 100 KB; a Node test loads the .wasm and reproduces the native surprise values to 1e-5 | [x] |
| 3 | Web app shell | Vite + TypeScript in `web/`; loads WASM; text-pattern and clickable piano input feed the model; expectation bars update per symbol; `npm run build` produces a static `dist` | [x] |
| 4 | Surprise timeline + Pages | Canvas timeline of surprise with anomaly highlights; "dream" mode generates continuations; `pages.yml` deploys; live link in README; empty state and landing section present | [ ] |
| 5 | Two-level hierarchy | Level-2 layer predicts level-1 state; test: on a pattern with period-16 structure, 2-level mean surprise < 1-level after equal training; UI toggle | [ ] |
| 6 | Quality | Playwright e2e for train, break pattern -> spike, dream; light/dark theme, keyboard shortcuts, mobile layout; benchmark with budget (>= 20k symbols/s native) enforced in CI | [ ] |
| 7 | Polish + release | Save/load model JSON round-trips exactly (test); presets (scale, lullaby, random); screenshots in README; error/loading states; v0.1.0 tag notes | [ ] |

## Stretch
- Learn rate/precision sliders with live free-energy readout
- MIDI input and audio playback of the dreamed melody
