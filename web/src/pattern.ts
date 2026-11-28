// Pure helpers: note names <-> symbol indices and text-pattern parsing.

export const NOTES = ["C", "D", "E", "F", "G", "A", "B"] as const;

export interface ParseResult {
  symbols: number[];
  /** Characters that were not understood (reported to the user, not silently dropped). */
  rejected: string[];
}

/**
 * Parse "C E G E", "ceg e" or "1 3 5 3" (scale degrees 1-7) into symbol indices.
 * Whitespace, commas, dashes and pipes are separators.
 */
export function parsePattern(text: string): ParseResult {
  const symbols: number[] = [];
  const rejected: string[] = [];
  for (const ch of text) {
    if (/[\s,\-|]/.test(ch)) continue;
    const letter = NOTES.indexOf(ch.toUpperCase() as (typeof NOTES)[number]);
    if (letter >= 0) symbols.push(letter);
    else if (ch >= "1" && ch <= "7") symbols.push(Number(ch) - 1);
    else rejected.push(ch);
  }
  return { symbols, rejected };
}

export function formatPattern(symbols: readonly number[]): string {
  return symbols.map((s) => NOTES[s] ?? "?").join(" ");
}

/** Index of the most likely symbol. */
export function argmax(p: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  return best;
}

export const PRESETS: Record<string, string> = {
  "Arpeggio": "C E G E C E G B",
  "Scale": "C D E F G A B A G F E D",
  "Twinkle": "C C G G A A G F F E E D D C",
  "Lullaby": "E E G E E G E G C B A A G D F E D C",
};

/** A random pattern of `len` notes (never the same note twice in a row). `rand` returns [0, 1). */
export function randomPattern(len = 8, rand: () => number = Math.random): string {
  const out: number[] = [];
  for (let i = 0; i < len; i++) {
    let n = Math.min(NOTES.length - 1, Math.floor(rand() * NOTES.length));
    if (i > 0 && n === out[i - 1]) n = (n + 1) % NOTES.length;
    out.push(n);
  }
  return out.map((n) => NOTES[n]).join(" ");
}
