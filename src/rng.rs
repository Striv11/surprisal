//! Tiny deterministic PRNG (xorshift64*), so training is reproducible
//! in native tests and in the browser with no dependencies.

#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point.
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [-scale, scale).
    pub fn symmetric(&mut self, scale: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * scale
    }

    /// Sample an index from a probability vector.
    pub fn sample(&mut self, probs: &[f32]) -> usize {
        let mut r = self.next_f32();
        for (i, p) in probs.iter().enumerate() {
            if r < *p {
                return i;
            }
            r -= p;
        }
        probs.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            let x = a.next_f32();
            assert_eq!(x, b.next_f32());
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn different_seeds_differ() {
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }

    #[test]
    fn sample_respects_distribution() {
        let mut r = Rng::new(3);
        let p = [0.0, 1.0, 0.0];
        for _ in 0..50 {
            assert_eq!(r.sample(&p), 1);
        }
    }
}
