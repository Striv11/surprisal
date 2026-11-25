import test from "node:test";
import assert from "node:assert/strict";
import { findAnomalies } from "../src/timeline.ts";

test("flags a spike after a learned, low-surprise stretch", () => {
  assert.deepEqual(findAnomalies([0.05, 0.04, 0.05, 4.2, 0.06]), [3]);
});

test("does not flag small fluctuations below the absolute floor", () => {
  assert.deepEqual(findAnomalies([0.01, 0.01, 0.4, 0.02]), []);
});

test("a uniformly high, unlearned stream is only flagged at the first symbol (no baseline)", () => {
  assert.deepEqual(findAnomalies([1.9, 2.0, 1.95, 2.1, 1.9]), [0]);
});

test("empty input", () => assert.deepEqual(findAnomalies([]), []));
