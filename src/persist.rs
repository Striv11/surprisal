//! Save/load a [`Model`] as JSON, with no dependencies.
//!
//! Every `f32` is stored as its raw IEEE-754 bit pattern (a `u32`), so a
//! save/load round trip is exact: a restored model continues bit-for-bit
//! like the original. The format is versioned and the loader validates sizes,
//! so a corrupt or hand-edited file produces an `Err`, never a panic.

use crate::model::{Config, Model};

const FORMAT: &str = "surprisal-model";
const VERSION: f64 = 1.0;

fn bits(v: &[f32]) -> String {
    let items: Vec<String> = v.iter().map(|x| x.to_bits().to_string()).collect();
    format!("[{}]", items.join(","))
}

impl Model {
    /// Serialise weights, state and config to a JSON string.
    pub fn to_json(&self) -> String {
        let c = self.config();
        let prev = self.prev_symbol().map_or(-1i64, |p| p as i64);
        let f = |x: f32| x.to_bits();
        format!(
            concat!(
                "{{\"format\":\"{}\",\"version\":{},\"vocab\":{},\"hidden\":{},",
                "\"infer_steps\":{},\"infer_rate\":{},\"learn_rate\":{},",
                "\"prior_precision\":{},\"top_hidden\":{},\"top_leak\":{},\"top_lr\":{},",
                "\"seed\":\"{}\",\"prev\":{},\"a\":{},\"b\":{},\"c\":{},\"d\":{},",
                "\"e\":{},\"f\":{},\"s\":{},\"z\":{}}}"
            ),
            FORMAT,
            VERSION as u32,
            c.vocab,
            c.hidden,
            c.infer_steps,
            f(c.infer_rate),
            f(c.learn_rate),
            f(c.prior_precision),
            c.top_hidden,
            f(c.top_leak),
            f(c.top_lr),
            c.seed,
            prev,
            bits(&self.a),
            bits(&self.b),
            bits(&self.c),
            bits(&self.d),
            bits(&self.e),
            bits(&self.f),
            bits(&self.s),
            bits(&self.z),
        )
    }

    /// Restore a model written by [`Model::to_json`].
    pub fn from_json(text: &str) -> Result<Model, String> {
        let root = Parser::new(text).parse_document()?;
        let obj = match &root {
            Json::Obj(o) => o,
            _ => return Err("expected a JSON object".into()),
        };
        let get = |k: &str| {
            obj.iter()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v)
                .ok_or_else(|| format!("missing field \"{k}\""))
        };
        if !matches!(get("format")?, Json::Str(s) if s == FORMAT) {
            return Err("not a surprisal model file".into());
        }
        if get("version")?.as_num()? != VERSION {
            return Err("unsupported model version".into());
        }
        let uint = |k: &str| -> Result<usize, String> {
            let n = get(k)?.as_num()?;
            if n < 0.0 || n.fract() != 0.0 || n > 1e9 {
                return Err(format!(
                    "field \"{k}\" must be a small non-negative integer"
                ));
            }
            Ok(n as usize)
        };
        let float = |k: &str| -> Result<f32, String> {
            let n = get(k)?.as_num()?;
            if n < 0.0 || n.fract() != 0.0 || n > u32::MAX as f64 {
                return Err(format!("field \"{k}\" is not an f32 bit pattern"));
            }
            Ok(f32::from_bits(n as u32))
        };
        let vocab = uint("vocab")?;
        let hidden = uint("hidden")?;
        let top = uint("top_hidden")?;
        if !(2..=256).contains(&vocab) || !(1..=1024).contains(&hidden) || top > 1024 {
            return Err("model dimensions out of range".into());
        }
        let seed = match get("seed")? {
            Json::Str(s) => s.parse::<u64>().map_err(|_| "bad seed".to_string())?,
            _ => return Err("seed must be a string".into()),
        };
        let cfg = Config {
            vocab,
            hidden,
            infer_steps: uint("infer_steps")?.min(1000),
            infer_rate: float("infer_rate")?,
            learn_rate: float("learn_rate")?,
            prior_precision: float("prior_precision")?,
            top_hidden: top,
            top_leak: float("top_leak")?,
            top_lr: float("top_lr")?,
            seed,
        };
        let prev = get("prev")?.as_num()?;
        let prev = if prev < 0.0 {
            None
        } else if prev.fract() == 0.0 && (prev as usize) < vocab {
            Some(prev as usize)
        } else {
            return Err("\"prev\" outside the vocabulary".into());
        };
        let arr = |k: &str, len: usize| -> Result<Vec<f32>, String> {
            let items = match get(k)? {
                Json::Arr(a) => a,
                _ => return Err(format!("field \"{k}\" must be an array")),
            };
            if items.len() != len {
                return Err(format!(
                    "field \"{k}\" has {} values, expected {len}",
                    items.len()
                ));
            }
            items
                .iter()
                .map(|j| {
                    let n = j.as_num()?;
                    if n < 0.0 || n.fract() != 0.0 || n > u32::MAX as f64 {
                        return Err(format!("field \"{k}\" holds a non-bit-pattern value"));
                    }
                    Ok(f32::from_bits(n as u32))
                })
                .collect()
        };
        let mut m = Model::new(Config {
            // Build a minimal skeleton, then overwrite every learned array.
            top_hidden: top,
            ..cfg
        });
        m.a = arr("a", hidden * hidden)?;
        m.b = arr("b", hidden * vocab)?;
        m.c = arr("c", vocab * hidden)?;
        m.d = arr("d", hidden * top)?;
        m.e = arr("e", top * top)?;
        m.f = arr("f", top * hidden)?;
        m.s = arr("s", hidden)?;
        m.z = arr("z", top)?;
        m.prev = prev;
        Ok(m)
    }
}

// ---- minimal JSON reader: objects, arrays, numbers, strings, null/bool ----

enum Json {
    Obj(Vec<(String, Json)>),
    Arr(Vec<Json>),
    Num(f64),
    Str(String),
    Other,
}

impl Json {
    fn as_num(&self) -> Result<f64, String> {
        match self {
            Json::Num(n) => Ok(*n),
            _ => Err("expected a number".into()),
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Parser {
            b: s.as_bytes(),
            i: 0,
            depth: 0,
        }
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn parse_document(mut self) -> Result<Json, String> {
        let v = self.value()?;
        self.ws();
        if self.i != self.b.len() {
            return Err("trailing data after JSON".into());
        }
        Ok(v)
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.b.get(self.i) {
            Some(b'{') => self.nest(|p| {
                p.i += 1;
                let mut items = Vec::new();
                p.ws();
                if p.b.get(p.i) == Some(&b'}') {
                    p.i += 1;
                    return Ok(Json::Obj(items));
                }
                loop {
                    p.ws();
                    let k = p.string()?;
                    p.eat(b':')?;
                    items.push((k, p.value()?));
                    p.ws();
                    match p.b.get(p.i) {
                        Some(b',') => p.i += 1,
                        Some(b'}') => {
                            p.i += 1;
                            return Ok(Json::Obj(items));
                        }
                        _ => return Err("expected ',' or '}'".into()),
                    }
                }
            }),
            Some(b'[') => self.nest(|p| {
                p.i += 1;
                let mut items = Vec::new();
                p.ws();
                if p.b.get(p.i) == Some(&b']') {
                    p.i += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(p.value()?);
                    p.ws();
                    match p.b.get(p.i) {
                        Some(b',') => p.i += 1,
                        Some(b']') => {
                            p.i += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err("expected ',' or ']'".into()),
                    }
                }
            }),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(c) if c.is_ascii_digit() || *c == b'-' => {
                let start = self.i;
                while self.i < self.b.len()
                    && matches!(
                        self.b[self.i],
                        b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
                    )
                {
                    self.i += 1;
                }
                std::str::from_utf8(&self.b[start..self.i])
                    .ok()
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(Json::Num)
                    .ok_or_else(|| format!("bad number at byte {start}"))
            }
            Some(_) => {
                for lit in ["true", "false", "null"] {
                    if self.b[self.i..].starts_with(lit.as_bytes()) {
                        self.i += lit.len();
                        return Ok(Json::Other);
                    }
                }
                Err(format!("unexpected character at byte {}", self.i))
            }
            None => Err("unexpected end of input".into()),
        }
    }

    fn nest(&mut self, f: impl FnOnce(&mut Self) -> Result<Json, String>) -> Result<Json, String> {
        self.depth += 1;
        if self.depth > 8 {
            return Err("JSON nested too deeply".into());
        }
        let r = f(self);
        self.depth -= 1;
        r
    }

    /// Strings in this format are plain ASCII; escapes are not supported.
    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.i) != Some(&b'"') {
            return Err(format!("expected string at byte {}", self.i));
        }
        self.i += 1;
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i] != b'"' {
            if self.b[self.i] == b'\\' {
                return Err("string escapes are not supported".into());
            }
            self.i += 1;
        }
        if self.i >= self.b.len() {
            return Err("unterminated string".into());
        }
        let s = String::from_utf8_lossy(&self.b[start..self.i]).into_owned();
        self.i += 1;
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained(top: usize) -> Model {
        let mut cfg = Config::new(6);
        cfg.top_hidden = top;
        cfg.seed = 9;
        let mut m = Model::new(cfg);
        let seq = [0usize, 2, 4, 2, 0, 2, 4, 5];
        for _ in 0..30 {
            m.reset_context();
            seq.iter().for_each(|&s| {
                m.observe(s);
            });
        }
        m
    }

    #[test]
    fn round_trip_is_exact_and_continues_identically() {
        for top in [0, 8] {
            let mut a = trained(top);
            let json = a.to_json();
            let mut b = Model::from_json(&json).expect("load");
            assert_eq!(json, b.to_json(), "re-serialising must be identical");
            assert_eq!(a.predict(), b.predict());
            // Mid-sequence state is kept too: keep learning in lockstep.
            for s in [1usize, 3, 5, 0, 2] {
                assert_eq!(a.observe(s).surprise, b.observe(s).surprise);
            }
            assert_eq!(a.generate(12, 5), b.generate(12, 5));
        }
    }

    #[test]
    fn rejects_malformed_files_without_panicking() {
        let good = trained(0).to_json();
        let bad = [
            String::new(),
            "[]".into(),
            "{".into(),
            good.replace("surprisal-model", "other"),
            good.replace("\"version\":1", "\"version\":2"),
            good.replace("\"vocab\":6", "\"vocab\":7"), // sizes no longer match
            good.replace("\"vocab\":6", "\"vocab\":1"),
            good.replace("\"prev\":", "\"prev\":99,\"x\":"),
            good.replace("\"a\":[", "\"a\":[-1,"),
            good[..good.len() / 2].to_string(),
            format!("{good} trailing"),
            "[".repeat(100),
        ];
        for (i, text) in bad.iter().enumerate() {
            assert!(Model::from_json(text).is_err(), "case {i} was accepted");
        }
    }
}
