//! Prints per-step surprise for a fixed melody (used by the Node parity test).
use surprisal::{Config, Model};

fn main() {
    let mut cfg = Config::new(7);
    cfg.seed = 42;
    let mut m = Model::new(cfg);
    let melody = [0usize, 2, 4, 2, 0, 2, 4, 6];
    for _ in 0..30 {
        m.reset_context();
        for &s in &melody {
            m.observe(s);
        }
    }
    m.reset_context();
    for &s in melody.iter().chain(&[3usize, 1]) {
        println!("{:.9}", m.observe(s).surprise);
    }
    let g: Vec<String> = m.generate(8, 5).iter().map(|x| x.to_string()).collect();
    println!("gen {}", g.join(" "));
}
