import test from "node:test";
import assert from "node:assert/strict";
import { parsePattern, formatPattern, argmax, PRESETS, randomPattern } from "../src/pattern.ts";

test("parses note names case-insensitively with separators", () => {
  assert.deepEqual(parsePattern("C e,G-a | b").symbols, [0, 2, 4, 5, 6]);
  assert.equal(parsePattern("ceg").symbols.length, 3);
});

test("parses scale degrees", () => {
  assert.deepEqual(parsePattern("1 3 5 7").symbols, [0, 2, 4, 6]);
});

test("reports rejected characters instead of dropping them silently", () => {
  const r = parsePattern("C x 8 D");
  assert.deepEqual(r.symbols, [0, 1]);
  assert.deepEqual(r.rejected, ["x", "8"]);
});

test("format round-trips", () => {
  assert.equal(formatPattern(parsePattern("C E G").symbols), "C E G");
});

test("argmax", () => {
  assert.equal(argmax([0.1, 0.7, 0.2]), 1);
});

test("all presets parse cleanly", () => {
  for (const [name, text] of Object.entries(PRESETS)) {
    const r = parsePattern(text);
    assert.equal(r.rejected.length, 0, name);
    assert.ok(r.symbols.length > 0, name);
  }
});

test("every preset parses cleanly and random patterns are valid", () => {
  for (const [name, text] of Object.entries(PRESETS)) {
    const r = parsePattern(text);
    assert.deepEqual(r.rejected, [], name);
    assert.ok(r.symbols.length >= 8, name);
  }
  let x = 0.37;
  const rand = () => (x = (x * 9301 + 0.49297) % 1);
  const p = randomPattern(12, rand);
  const { symbols, rejected } = parsePattern(p);
  assert.equal(symbols.length, 12);
  assert.deepEqual(rejected, []);
  assert.ok(symbols.every((n, i) => i === 0 || n !== symbols[i - 1]));
  assert.equal(randomPattern(5, () => 0).split(" ").length, 5); // constant rand still avoids repeats
});
