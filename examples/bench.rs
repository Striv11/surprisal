//! Throughput benchmark with a budget. Run: `cargo run --release --example bench`.
//! Exits non-zero if either configuration processes fewer than BUDGET symbols/s.
//! Override the budget with SURPRISAL_BENCH_BUDGET (symbols/s) on slow machines.
use std::time::Instant;
use surprisal::{Config, Model};

const DEFAULT_BUDGET: f64 = 20_000.0;
const SYMBOLS: usize = 60_000;

fn run(name: &str, cfg: Config) -> f64 {
    let vocab = cfg.vocab;
    let mut model = Model::new(cfg);
    let tune = [0usize, 2, 4, 2, 0, 2, 4, 6];
    // Warm up so caches and branch predictors are settled.
    for i in 0..2_000 {
        model.observe(tune[i % tune.len()] % vocab);
    }
    let start = Instant::now();
    let mut sink = 0.0f32;
    for i in 0..SYMBOLS {
        sink += model.observe(tune[i % tune.len()] % vocab).surprise;
    }
    let secs = start.elapsed().as_secs_f64();
    let rate = SYMBOLS as f64 / secs;
    println!("{name:<10} {rate:>10.0} symbols/s  (checksum {sink:.3})");
    rate
}

fn main() {
    let budget = std::env::var("SURPRISAL_BENCH_BUDGET")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(DEFAULT_BUDGET);
    let one = run("one-level", Config::new(7));
    let mut two_cfg = Config::new(7);
    two_cfg.top_hidden = 12;
    let two = run("two-level", two_cfg);
    println!("budget     {budget:>10.0} symbols/s");
    if one < budget || two < budget {
        eprintln!("FAIL: throughput below budget");
        std::process::exit(1);
    }
}
