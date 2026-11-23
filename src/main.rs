//! Terminal demo: learn a melody online and watch surprise fall.
use surprisal::{Config, Model};

const NAMES: [&str; 7] = ["C", "D", "E", "F", "G", "A", "B"];

fn main() {
    let tune = [0usize, 2, 4, 2, 0, 2, 4, 6];
    let seq: Vec<usize> = tune.iter().cycle().take(64).cloned().collect();
    let mut model = Model::new(Config::new(NAMES.len()));
    println!("epoch  mean surprise (nats)  next-note accuracy");
    for epoch in 0..=120 {
        model.reset_context();
        let (mut total, mut hits) = (0.0, 0);
        for &n in &seq {
            let r = model.observe(n);
            total += r.surprise;
            hits += r.correct as usize;
        }
        if epoch % 20 == 0 {
            println!(
                "{epoch:>5}  {:>20.3}  {:>17.0}%",
                total / seq.len() as f32,
                100.0 * hits as f32 / seq.len() as f32
            );
        }
    }
    model.reset_context();
    for &n in &seq[..8] {
        model.observe(n);
    }
    let gen: Vec<&str> = model
        .generate(16, 42)
        .into_iter()
        .map(|i| NAMES[i])
        .collect();
    println!("dreamed continuation: {}", gen.join(" "));
}
