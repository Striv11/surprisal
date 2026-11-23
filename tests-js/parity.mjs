// Loads the release .wasm and checks it reproduces native surprise values.
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import assert from "node:assert/strict";

const wasmPath = "target/wasm32-unknown-unknown/release/surprisal.wasm";
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const x = instance.exports;
const size = readFileSync(wasmPath).length;
assert.ok(size < 100 * 1024, `wasm is ${size} bytes, budget 100 KB`);

const native = execFileSync("cargo", ["run", "--release", "--quiet", "--example", "trace"], { encoding: "utf8" })
  .trim().split("\n");
const nativeGen = native.pop().split(" ").slice(1).map(Number);
const nativeS = native.map(Number);

const h = x.sp_create(7, 24, 42);
assert.ok(h !== 0);
const melody = [0, 2, 4, 2, 0, 2, 4, 6];
for (let e = 0; e < 200; e++) { x.sp_reset(h); for (const s of melody) x.sp_observe(h, s); }
x.sp_reset(h);
const got = [...melody, 3, 1].map((s) => x.sp_observe(h, s));
got.forEach((v, i) => assert.ok(Math.abs(v - nativeS[i]) < 1e-5, `step ${i}: wasm ${v} vs native ${nativeS[i]}`));
assert.ok(got[8] > 5 * Math.max(...got.slice(1, 8)), "wrong note should spike");

const buf = x.sp_alloc(7 * 4);
assert.equal(x.sp_predict(h, buf, 7), 7);
const p = new Float32Array(x.memory.buffer, buf, 7);
assert.ok(Math.abs(p.reduce((a, b) => a + b) - 1) < 1e-5);
x.sp_dealloc(buf, 28);

const ob = x.sp_alloc(8 * 4);
assert.equal(x.sp_generate(h, 8, 5, ob), 8);
const gen = [...new Uint32Array(x.memory.buffer, ob, 8)];
assert.deepEqual(gen, nativeGen);
x.sp_dealloc(ob, 32);
x.sp_free(h);
console.log(`wasm parity ok (${size} bytes)`);
