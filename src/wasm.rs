//! Hand-written C ABI over [`Model`], used by the browser build (no wasm-bindgen).
//!
//! Handles are opaque pointers. Buffers cross the boundary through
//! `sp_alloc` / `sp_dealloc` so the host can read and write linear memory.
//! The module is compiled on every target so the exact same functions are
//! unit-tested natively.
#![allow(clippy::missing_safety_doc)]

use crate::model::{Config, Model};

/// Create a single-level model. Returns null for invalid sizes.
#[no_mangle]
pub extern "C" fn sp_create(vocab: u32, hidden: u32, seed: u32) -> *mut Model {
    sp_create2(vocab, hidden, 0, seed)
}

/// Create a model with an optional slow second level of `top` units
/// (0 = single level, at most 1024). Returns null for invalid sizes.
#[no_mangle]
pub extern "C" fn sp_create2(vocab: u32, hidden: u32, top: u32, seed: u32) -> *mut Model {
    if !(2..=256).contains(&vocab) || !(1..=1024).contains(&hidden) || top > 1024 {
        return std::ptr::null_mut();
    }
    let mut cfg = Config::new(vocab as usize);
    cfg.hidden = hidden as usize;
    cfg.top_hidden = top as usize;
    cfg.top_leak = 0.4;
    cfg.seed = seed as u64;
    Box::into_raw(Box::new(Model::new(cfg)))
}

#[no_mangle]
pub unsafe extern "C" fn sp_free(m: *mut Model) {
    if !m.is_null() {
        drop(Box::from_raw(m));
    }
}

/// Learn from one symbol; returns its surprise in nats, or -1 on bad input.
#[no_mangle]
pub unsafe extern "C" fn sp_observe(m: *mut Model, symbol: u32) -> f32 {
    match m.as_mut() {
        Some(m) if (symbol as usize) < m.config().vocab => m.observe(symbol as usize).surprise,
        _ => -1.0,
    }
}

/// Write the next-symbol distribution into `out` (capacity `cap` floats).
/// Returns the vocabulary size, or 0 if the buffer is too small / handle null.
#[no_mangle]
pub unsafe extern "C" fn sp_predict(m: *const Model, out: *mut f32, cap: u32) -> u32 {
    let Some(m) = m.as_ref() else { return 0 };
    let p = m.predict();
    if out.is_null() || (cap as usize) < p.len() {
        return 0;
    }
    std::ptr::copy_nonoverlapping(p.as_ptr(), out, p.len());
    p.len() as u32
}

/// Sample `len` symbols from the model's own predictions into `out` (u32s).
/// Does not change learned weights. Returns the number written.
#[no_mangle]
pub unsafe extern "C" fn sp_generate(m: *const Model, len: u32, seed: u32, out: *mut u32) -> u32 {
    let Some(m) = m.as_ref() else { return 0 };
    if out.is_null() {
        return 0;
    }
    let seq = m.generate(len as usize, seed as u64);
    for (i, s) in seq.iter().enumerate() {
        *out.add(i) = *s as u32;
    }
    seq.len() as u32
}

/// Forget the running context, keeping learned weights.
#[no_mangle]
pub unsafe extern "C" fn sp_reset(m: *mut Model) {
    if let Some(m) = m.as_mut() {
        m.reset_context();
    }
}

/// Serialise the model to JSON. Returns a buffer from `sp_alloc` (free it with
/// `sp_dealloc(ptr, len)`) and writes its length to `len_out`; null on failure.
#[no_mangle]
pub unsafe extern "C" fn sp_save(m: *const Model, len_out: *mut u32) -> *mut u8 {
    let (Some(m), false) = (m.as_ref(), len_out.is_null()) else {
        return std::ptr::null_mut();
    };
    let json = m.to_json();
    let p = sp_alloc(json.len() as u32);
    std::ptr::copy_nonoverlapping(json.as_ptr(), p, json.len());
    *len_out = json.len() as u32;
    p
}

/// Restore a model from `len` bytes of JSON written by `sp_save`.
/// Returns null if the data is invalid.
#[no_mangle]
pub unsafe extern "C" fn sp_load(data: *const u8, len: u32) -> *mut Model {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(data, len as usize);
    match std::str::from_utf8(bytes)
        .ok()
        .and_then(|t| Model::from_json(t).ok())
    {
        Some(m) => Box::into_raw(Box::new(m)),
        None => std::ptr::null_mut(),
    }
}

/// Vocabulary, hidden and top sizes of a model (for UI after a load).
#[no_mangle]
pub unsafe extern "C" fn sp_dims(m: *const Model, out: *mut u32) -> u32 {
    match (m.as_ref(), out.is_null()) {
        (Some(m), false) => {
            let c = m.config();
            *out = c.vocab as u32;
            *out.add(1) = c.hidden as u32;
            *out.add(2) = c.top_hidden as u32;
            3
        }
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn sp_alloc(bytes: u32) -> *mut u8 {
    let mut v = vec![0u8; bytes.max(1) as usize].into_boxed_slice();
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[no_mangle]
pub unsafe extern "C" fn sp_dealloc(p: *mut u8, bytes: u32) {
    if !p.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            p,
            bytes.max(1) as usize,
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_matches_direct_model() {
        let mut cfg = Config::new(5);
        cfg.hidden = 16;
        cfg.seed = 7;
        let mut direct = Model::new(cfg);
        unsafe {
            let h = sp_create(5, 16, 7);
            assert!(!h.is_null());
            let seq = [0usize, 2, 4, 2, 0, 2, 4, 2];
            for &s in &seq {
                let a = sp_observe(h, s as u32);
                assert_eq!(a, direct.observe(s).surprise);
            }
            let mut buf = [0f32; 5];
            assert_eq!(sp_predict(h, buf.as_mut_ptr(), 5), 5);
            assert!((buf.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert_eq!(buf.to_vec(), direct.predict());
            let mut out = [0u32; 6];
            assert_eq!(sp_generate(h, 6, 3, out.as_mut_ptr()), 6);
            let want: Vec<u32> = direct.generate(6, 3).iter().map(|&x| x as u32).collect();
            assert_eq!(out.to_vec(), want);
            sp_reset(h);
            sp_free(h);
        }
    }

    #[test]
    fn save_and_load_through_the_abi() {
        unsafe {
            let h = sp_create2(5, 12, 6, 3);
            for s in [0u32, 1, 2, 3, 0, 1, 2, 3] {
                sp_observe(h, s);
            }
            let mut len = 0u32;
            let p = sp_save(h, &mut len);
            assert!(!p.is_null() && len > 0);
            let g = sp_load(p, len);
            assert!(!g.is_null());
            let mut dims = [0u32; 3];
            assert_eq!(sp_dims(g, dims.as_mut_ptr()), 3);
            assert_eq!(dims, [5, 12, 6]);
            assert_eq!(sp_observe(h, 4), sp_observe(g, 4));
            assert!(sp_load(p, len / 2).is_null());
            assert!(sp_load(std::ptr::null(), 4).is_null());
            assert!(sp_save(h, std::ptr::null_mut()).is_null());
            sp_dealloc(p, len);
            sp_free(g);
            sp_free(h);
        }
    }

    #[test]
    fn rejects_bad_input() {
        unsafe {
            assert!(sp_create(1, 8, 1).is_null());
            assert!(sp_create(4, 0, 1).is_null());
            assert!(sp_create2(4, 8, 2000, 1).is_null());
            let t = sp_create2(4, 8, 8, 1);
            assert!(!t.is_null());
            assert!(sp_observe(t, 2) >= 0.0);
            sp_free(t);
            let h = sp_create(4, 8, 1);
            assert_eq!(sp_observe(h, 4), -1.0);
            assert_eq!(sp_observe(std::ptr::null_mut(), 0), -1.0);
            let mut small = [0f32; 2];
            assert_eq!(sp_predict(h, small.as_mut_ptr(), 2), 0);
            sp_free(h);
        }
    }

    #[test]
    fn alloc_roundtrip() {
        let p = sp_alloc(64);
        assert!(!p.is_null());
        unsafe { sp_dealloc(p, 64) };
    }
}
