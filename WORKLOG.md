# Work log

Newest entries at the bottom. Each entry records what actually changed, what was learned, and what comes next.

## 2026-11-23 (Day 1)
- Chose "predictive-coding sequence learner" from the Cognition backlog section (Rust, differing from recent projects); named it surprisal.
- Wrote the dependency-free Rust core: xorshift PRNG and a temporal predictive-coding network (prior, softmax prediction, iterative inference, local weight updates, generation).
- Tests: surprise falls >4x on a melody, spikes on an out-of-pattern note, deterministic training, generation reproduces a cycle, vocab bounds.
- Terminal demo shows mean surprise 1.45 -> 0.05 nats over 120 epochs.
- Learned: with a tanh prior and a cross-entropy readout the plain local rules converge quickly without tuning.
- Next: C-ABI WASM exports and a Node parity test (Day 2). Pages deploy and UI come later in the arc.
