/** Keyboard shortcut mapping, kept free of DOM types so it can be unit-tested in Node. */
export type KeyAction =
  | { kind: "note"; note: number }
  | { kind: "train" }
  | { kind: "feed" }
  | { kind: "dream" }
  | { kind: "reset" }
  | { kind: "theme" };

export interface KeyInfo {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  altKey?: boolean;
  /** Tag name of the focused element, if any (e.g. "INPUT"). */
  targetTag?: string;
}

const TYPING_TAGS = new Set(["INPUT", "TEXTAREA", "SELECT", "BUTTON"]);

/**
 * Map a key press to an action. Digits 1-7 play scale degrees; t/f/d/r/m trigger
 * Train, Feed, Dream, Reset, theme toggle. Ignored while typing in a field, when a
 * button has focus (Enter/Space belong to it) or when a modifier is held.
 */
export function keyAction(k: KeyInfo): KeyAction | null {
  if (k.ctrlKey || k.metaKey || k.altKey) return null;
  if (k.targetTag && TYPING_TAGS.has(k.targetTag.toUpperCase())) return null;
  if (/^[1-7]$/.test(k.key)) return { kind: "note", note: Number(k.key) - 1 };
  switch (k.key.toLowerCase()) {
    case "t": return { kind: "train" };
    case "f": return { kind: "feed" };
    case "d": return { kind: "dream" };
    case "r": return { kind: "reset" };
    case "m": return { kind: "theme" };
    default: return null;
  }
}
