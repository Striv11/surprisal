import { test } from "node:test";
import assert from "node:assert/strict";
import { keyAction } from "../src/keys.ts";
import { nextTheme, parseTheme } from "../src/theme.ts";

test("digits 1-7 map to notes 0-6", () => {
  assert.deepEqual(keyAction({ key: "1" }), { kind: "note", note: 0 });
  assert.deepEqual(keyAction({ key: "7" }), { kind: "note", note: 6 });
  assert.equal(keyAction({ key: "8" }), null);
  assert.equal(keyAction({ key: "0" }), null);
});

test("letter shortcuts are case-insensitive", () => {
  assert.deepEqual(keyAction({ key: "T" }), { kind: "train" });
  assert.deepEqual(keyAction({ key: "d" }), { kind: "dream" });
  assert.deepEqual(keyAction({ key: "r" }), { kind: "reset" });
  assert.deepEqual(keyAction({ key: "f" }), { kind: "feed" });
  assert.deepEqual(keyAction({ key: "m" }), { kind: "theme" });
  assert.equal(keyAction({ key: "x" }), null);
});

test("ignored while typing, on buttons, or with modifiers", () => {
  assert.equal(keyAction({ key: "1", targetTag: "INPUT" }), null);
  assert.equal(keyAction({ key: "d", targetTag: "textarea" }), null);
  assert.equal(keyAction({ key: "d", targetTag: "BUTTON" }), null);
  assert.equal(keyAction({ key: "r", ctrlKey: true }), null);
  assert.equal(keyAction({ key: "r", metaKey: true }), null);
  assert.deepEqual(keyAction({ key: "r", targetTag: "BODY" }), { kind: "reset" });
});

test("theme parsing and cycling", () => {
  assert.equal(parseTheme("dark"), "dark");
  assert.equal(parseTheme("light"), "light");
  assert.equal(parseTheme("bogus"), "auto");
  assert.equal(parseTheme(null), "auto");
  assert.equal(nextTheme(nextTheme(nextTheme("auto"))), "auto");
  assert.equal(nextTheme("auto"), "light");
});
