export type Theme = "auto" | "light" | "dark";
export const THEME_KEY = "surprisal-theme";

/** Parse a stored value, falling back to "auto" for anything unexpected. */
export function parseTheme(raw: string | null | undefined): Theme {
  return raw === "light" || raw === "dark" ? raw : "auto";
}

/** auto -> light -> dark -> auto */
export function nextTheme(t: Theme): Theme {
  return t === "auto" ? "light" : t === "light" ? "dark" : "auto";
}
