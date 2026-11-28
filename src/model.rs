//! Temporal predictive-coding network.
//!
//! Generative model, with hidden state `s_t` (size H) and symbols `o_t` (vocab V):
//!
//! ```text
//! prior      mu_t = tanh(A s_{t-1} + B onehot(o_{t-1}))
//! prediction p_t  = softmax(C mu_t)            <- made BEFORE seeing o_t
//! ```
//!
//! When `o_t` arrives we compute surprise `-ln p_t[o_t]`, then run `K` steps of
//! iterative inference that move `s_t` to reduce two prediction errors:
//! the sensory error `e_o = onehot(o_t) - softmax(C s_t)` and the
//! state error `e_s = s_t - mu_t`. Weights then change with purely local
//! rules (error x presynaptic activity), as in predictive-coding theory.

use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab: usize,
    pub hidden: usize,
    /// Inference steps per symbol.
    pub infer_steps: usize,
    /// Inference step size.
    pub infer_rate: f32,
    /// Weight learning rate.
    pub learn_rate: f32,
    /// Precision (weight) of the state prior relative to sensory evidence.
    pub prior_precision: f32,
    /// Size of the slow second level; 0 disables it (single-level model).
    pub top_hidden: usize,
    /// Leak of the level-2 state per symbol (smaller = slower timescale).
    pub top_leak: f32,
    /// Learning-rate multiplier for the top-down weights.
    pub top_lr: f32,
    pub seed: u64,
}

impl Config {
    pub fn new(vocab: usize) -> Self {
        Config {
            vocab,
            hidden: 24,
            infer_steps: 8,
            infer_rate: 0.3,
            learn_rate: 0.05,
            prior_precision: 0.5,
            top_hidden: 0,
            top_leak: 0.25,
            top_lr: 10.0,
            seed: 1,
        }
    }
}

/// Everything the UI wants to know about one observed symbol.
#[derive(Clone, Debug)]
pub struct StepReport {
    /// Probability distribution the model predicted for this symbol.
    pub predicted: Vec<f32>,
    /// Surprise in nats: -ln p(observed).
    pub surprise: f32,
    /// Whether the most probable prediction matched.
    pub correct: bool,
}

#[derive(Clone, Debug)]
pub struct Model {
    cfg: Config,
    pub(crate) a: Vec<f32>, // H x H
    pub(crate) b: Vec<f32>, // H x V
    pub(crate) c: Vec<f32>, // V x H
    pub(crate) s: Vec<f32>, // current state (H)
    pub(crate) d: Vec<f32>, // H x T top-down weights (level 2 -> level-1 prior), learned
    pub(crate) e: Vec<f32>, // T x T level-2 recurrence (fixed random)
    pub(crate) f: Vec<f32>, // T x H level-1 state -> level 2 (fixed random)
    pub(crate) z: Vec<f32>, // level-2 state (T), a slow summary of level-1 states
    pub(crate) prev: Option<usize>,
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let z: f32 = out.iter().sum();
    out.iter_mut().for_each(|x| *x /= z);
    out
}

impl Model {
    pub fn new(cfg: Config) -> Self {
        assert!(cfg.vocab >= 2 && cfg.hidden >= 1, "vocab >= 2, hidden >= 1");
        let mut rng = Rng::new(cfg.seed);
        let (h, v) = (cfg.hidden, cfg.vocab);
        let sa = 1.0 / (h as f32).sqrt();
        let a = (0..h * h).map(|_| rng.symmetric(sa)).collect();
        let b = (0..h * v).map(|_| rng.symmetric(0.5)).collect();
        let c = (0..v * h).map(|_| rng.symmetric(sa)).collect();
        let t = cfg.top_hidden;
        let st = 1.0 / (t.max(1) as f32).sqrt();
        let d = vec![0.0; h * t];
        let e = (0..t * t).map(|_| rng.symmetric(st)).collect();
        let f = (0..t * h).map(|_| rng.symmetric(sa)).collect();
        Model {
            d,
            e,
            f,
            z: vec![0.0; t],
            s: vec![0.0; h],
            a,
            b,
            c,
            prev: None,
            cfg,
        }
    }

    pub(crate) fn prev_symbol(&self) -> Option<usize> {
        self.prev
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Forget the running context (start of a new sequence) but keep weights.
    pub fn reset_context(&mut self) {
        self.s.iter_mut().for_each(|x| *x = 0.0);
        self.z.iter_mut().for_each(|x| *x = 0.0);
        self.prev = None;
    }

    /// Prior over the next hidden state given the context so far.
    fn prior(&self) -> Vec<f32> {
        let (h, v) = (self.cfg.hidden, self.cfg.vocab);
        (0..h)
            .map(|i| {
                let mut x: f32 = self.a[i * h..(i + 1) * h]
                    .iter()
                    .zip(&self.s)
                    .map(|(w, s)| w * s)
                    .sum();
                if let Some(p) = self.prev {
                    x += self.b[i * v + p];
                }
                let t = self.z.len();
                x += self.d[i * t..(i + 1) * t]
                    .iter()
                    .zip(&self.z)
                    .map(|(w, z)| w * z)
                    .sum::<f32>();
                x.tanh()
            })
            .collect()
    }

    fn readout(&self, state: &[f32]) -> Vec<f32> {
        let h = self.cfg.hidden;
        let logits: Vec<f32> = (0..self.cfg.vocab)
            .map(|k| {
                self.c[k * h..(k + 1) * h]
                    .iter()
                    .zip(state)
                    .map(|(w, s)| w * s)
                    .sum()
            })
            .collect();
        softmax(&logits)
    }

    /// Distribution over the next symbol, without learning.
    pub fn predict(&self) -> Vec<f32> {
        self.readout(&self.prior())
    }

    /// Observe a symbol: report the prediction/surprise, infer, then learn.
    pub fn observe(&mut self, symbol: usize) -> StepReport {
        let (h, v) = (self.cfg.hidden, self.cfg.vocab);
        assert!(symbol < v, "symbol out of vocabulary");
        let mu = self.prior();
        let predicted = self.readout(&mu);
        let surprise = -predicted[symbol].max(1e-9).ln();
        let argmax =
            predicted.iter().enumerate().fold(
                0,
                |best, (i, p)| if *p > predicted[best] { i } else { best },
            );

        // Iterative inference on the hidden state.
        let mut s = mu.clone();
        let lam = self.cfg.prior_precision;
        for _ in 0..self.cfg.infer_steps {
            let p = self.readout(&s);
            for i in 0..h {
                let mut drive = 0.0;
                for k in 0..v {
                    let e = (if k == symbol { 1.0 } else { 0.0 }) - p[k];
                    drive += self.c[k * h + i] * e;
                }
                s[i] += self.cfg.infer_rate * (drive - lam * (s[i] - mu[i]));
            }
        }

        // Local weight updates.
        let p = self.readout(&s);
        let lr = self.cfg.learn_rate;
        for k in 0..v {
            let e = (if k == symbol { 1.0 } else { 0.0 }) - p[k];
            for i in 0..h {
                self.c[k * h + i] += lr * e * s[i];
            }
        }
        for i in 0..h {
            let es = (s[i] - mu[i]) * (1.0 - mu[i] * mu[i]);
            let g = lr * lam * es;
            for j in 0..h {
                self.a[i * h + j] += g * self.s[j];
            }
            if let Some(pv) = self.prev {
                self.b[i * v + pv] += g;
            }
            // Level 2 predicts the level-1 state: same local rule, presynaptic = z.
            let t = self.z.len();
            for j in 0..t {
                self.d[i * t + j] += self.cfg.top_lr * g * self.z[j];
            }
        }
        // Level 2 integrates the inferred level-1 state on a slower timescale.
        let t = self.z.len();
        if t > 0 {
            let leak = self.cfg.top_leak;
            let next: Vec<f32> = (0..t)
                .map(|k| {
                    let rec: f32 = self.e[k * t..(k + 1) * t]
                        .iter()
                        .zip(&self.z)
                        .map(|(w, z)| w * z)
                        .sum();
                    let inp: f32 = self.f[k * h..(k + 1) * h]
                        .iter()
                        .zip(&s)
                        .map(|(w, x)| w * x)
                        .sum();
                    (1.0 - leak) * self.z[k] + leak * (rec + inp).tanh()
                })
                .collect();
            self.z = next;
        }

        self.s = s;
        self.prev = Some(symbol);
        StepReport {
            predicted,
            surprise,
            correct: argmax == symbol,
        }
    }

    /// Sample a continuation by feeding the model its own predictions.
    /// Does not change learned weights.
    pub fn generate(&self, len: usize, seed: u64) -> Vec<usize> {
        let mut m = self.clone();
        m.cfg.learn_rate = 0.0;
        let mut rng = Rng::new(seed);
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let sym = rng.sample(&m.predict());
            m.observe(sym);
            out.push(sym);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean_surprise(m: &mut Model, seq: &[usize]) -> f32 {
        m.reset_context();
        seq.iter().map(|&s| m.observe(s).surprise).sum::<f32>() / seq.len() as f32
    }

    #[test]
    fn softmax_is_a_distribution() {
        let p = softmax(&[1.0, 2.0, 3.0, -50.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn untrained_model_is_near_uniform() {
        let m = Model::new(Config::new(8));
        let p = m.predict();
        assert_eq!(p.len(), 8);
        assert!(p.iter().all(|x| (x - 0.125).abs() < 0.1));
    }

    #[test]
    fn learns_a_repeating_melody() {
        let melody: Vec<usize> = [0, 2, 4, 2, 0, 2, 4, 5]
            .iter()
            .cycle()
            .take(64)
            .cloned()
            .collect();
        let mut m = Model::new(Config::new(6));
        let before = mean_surprise(&mut m, &melody);
        for _ in 0..150 {
            mean_surprise(&mut m, &melody);
        }
        let after = mean_surprise(&mut m, &melody);
        assert!(after < before * 0.25, "before {before}, after {after}");
    }

    #[test]
    fn surprise_spikes_on_a_wrong_note() {
        let pattern = [0usize, 1, 2, 3];
        let seq: Vec<usize> = pattern.iter().cycle().take(40).cloned().collect();
        let mut m = Model::new(Config::new(5));
        for _ in 0..200 {
            mean_surprise(&mut m, &seq);
        }
        m.reset_context();
        let mut expected = 0.0;
        for &s in &seq[..10] {
            expected = m.observe(s).surprise;
        }
        let anomaly = m.observe(4).surprise; // pattern says 2 comes next
        assert!(anomaly > expected * 5.0, "anomaly {anomaly} vs {expected}");
    }

    #[test]
    fn training_is_deterministic() {
        let seq = [0usize, 1, 0, 2, 0, 1, 0, 2];
        let run = || {
            let mut m = Model::new(Config::new(3));
            for _ in 0..10 {
                mean_surprise(&mut m, &seq);
            }
            m.predict()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn generation_reproduces_learned_pattern() {
        let seq: Vec<usize> = [0usize, 1, 2, 3].iter().cycle().take(48).cloned().collect();
        let mut m = Model::new(Config::new(4));
        for _ in 0..300 {
            mean_surprise(&mut m, &seq);
        }
        m.reset_context();
        for &s in &seq[..8] {
            m.observe(s);
        }
        let gen = m.generate(8, 9);
        let ok = gen.windows(2).filter(|w| w[1] == (w[0] + 1) % 4).count();
        assert!(ok >= 6, "generated {gen:?}");
    }

    #[test]
    #[should_panic]
    fn rejects_out_of_vocab() {
        Model::new(Config::new(3)).observe(3);
    }

    /// Period-16 tune whose two halves share every local context and differ only
    /// at the very end: telling them apart needs a slower timescale.
    fn period16() -> Vec<usize> {
        let mut seq = Vec::new();
        for _ in 0..8 {
            seq.extend_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3]);
            seq.extend_from_slice(&[0, 1, 2, 3, 0, 1, 2, 4]);
        }
        seq
    }

    fn trained_surprise(top_hidden: usize, seed: u64) -> f32 {
        let seq = period16();
        let mut cfg = Config::new(5);
        cfg.top_hidden = top_hidden;
        cfg.top_leak = 0.4;
        cfg.seed = seed;
        let mut m = Model::new(cfg);
        for _ in 0..300 {
            mean_surprise(&mut m, &seq);
        }
        mean_surprise(&mut m, &seq)
    }

    #[test]
    fn two_levels_beat_one_on_period16_structure() {
        for seed in 1..=3 {
            let one = trained_surprise(0, seed);
            let two = trained_surprise(32, seed);
            assert!(two < one * 0.6, "seed {seed}: one {one}, two {two}");
        }
    }

    #[test]
    fn two_level_training_is_deterministic_and_resets() {
        let a = trained_surprise(16, 7);
        assert_eq!(a, trained_surprise(16, 7));
        let mut cfg = Config::new(3);
        cfg.top_hidden = 8;
        let mut m = Model::new(cfg);
        let fresh = m.predict();
        m.observe(1);
        m.reset_context();
        assert_eq!(fresh, m.predict());
    }
}
