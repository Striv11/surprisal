//! surprisal: a predictive-coding sequence learner.
pub mod model;
pub mod persist;
pub mod rng;
pub mod wasm;

pub use model::{Config, Model, StepReport};
