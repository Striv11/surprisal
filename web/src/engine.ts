// Thin typed wrapper over the hand-written C ABI in src/wasm.rs.

interface Exports {
  memory: WebAssembly.Memory;
  sp_create2(vocab: number, hidden: number, top: number, seed: number): number;
  sp_free(h: number): void;
  sp_observe(h: number, symbol: number): number;
  sp_predict(h: number, out: number, cap: number): number;
  sp_generate(h: number, len: number, seed: number, out: number): number;
  sp_reset(h: number): void;
  sp_save(h: number, lenOut: number): number;
  sp_load(data: number, len: number): number;
  sp_dims(h: number, out: number): number;
  sp_alloc(bytes: number): number;
  sp_dealloc(p: number, bytes: number): void;
}

export class Engine {
  private handle: number;
  private x: Exports;
  readonly vocab: number;
  readonly hidden: number;
  readonly top: number;
  readonly seed: number;

  private constructor(x: Exports, vocab: number, hidden: number, top: number, seed: number, handle?: number) {
    this.x = x;
    this.vocab = vocab;
    this.hidden = hidden;
    this.top = top;
    this.seed = seed;
    this.handle = handle ?? x.sp_create2(vocab, hidden, top, seed);
    if (this.handle === 0) throw new Error("sp_create2 rejected the configuration");
  }

  static async load(
    wasm: BufferSource | Response | Promise<Response>,
    vocab = 7,
    hidden = 24,
    seed = 42,
    top = 0,
  ): Promise<Engine> {
    let instance: WebAssembly.Instance;
    if (wasm instanceof Response || wasm instanceof Promise) {
      ({ instance } = await WebAssembly.instantiateStreaming(wasm, {}));
    } else {
      ({ instance } = await WebAssembly.instantiate(wasm, {}));
    }
    return new Engine(instance.exports as unknown as Exports, vocab, hidden, top, seed);
  }

  /** A fresh, untrained engine with the same wasm instance but a different level count. */
  withTop(top: number): Engine {
    return new Engine(this.x, this.vocab, this.hidden, top, this.seed);
  }

  /** Serialise weights and state to JSON (floats stored as exact bit patterns). */
  save(): string {
    this.check();
    const lenPtr = this.x.sp_alloc(4);
    try {
      const p = this.x.sp_save(this.handle, lenPtr);
      if (p === 0) throw new Error("sp_save failed");
      const len = new Uint32Array(this.x.memory.buffer, lenPtr, 1)[0];
      try {
        return new TextDecoder().decode(new Uint8Array(this.x.memory.buffer, p, len));
      } finally {
        this.x.sp_dealloc(p, len);
      }
    } finally {
      this.x.sp_dealloc(lenPtr, 4);
    }
  }

  /** A new engine (same wasm instance) restored from `save()` output. Throws on invalid data. */
  restore(json: string): Engine {
    const bytes = new TextEncoder().encode(json);
    const p = this.x.sp_alloc(bytes.length);
    let handle: number;
    try {
      new Uint8Array(this.x.memory.buffer, p, bytes.length).set(bytes);
      handle = this.x.sp_load(p, bytes.length);
    } finally {
      this.x.sp_dealloc(p, bytes.length);
    }
    if (handle === 0) throw new Error("not a valid surprisal model file");
    const out = this.x.sp_alloc(12);
    try {
      this.x.sp_dims(handle, out);
      const [v, h, t] = new Uint32Array(this.x.memory.buffer, out, 3);
      return new Engine(this.x, v, h, t, this.seed, handle);
    } finally {
      this.x.sp_dealloc(out, 12);
    }
  }

  /** Learn one symbol; returns its surprise in nats (computed before learning). */
  observe(symbol: number): number {
    this.check();
    const s = this.x.sp_observe(this.handle, symbol);
    if (s < 0) throw new RangeError(`symbol ${symbol} outside 0..${this.vocab - 1}`);
    return s;
  }

  /** Distribution over the next symbol. */
  predict(): Float32Array {
    this.check();
    const bytes = this.vocab * 4;
    const p = this.x.sp_alloc(bytes);
    try {
      if (this.x.sp_predict(this.handle, p, this.vocab) !== this.vocab) throw new Error("sp_predict failed");
      // Copy: wasm memory may grow and detach views.
      return new Float32Array(this.x.memory.buffer, p, this.vocab).slice();
    } finally {
      this.x.sp_dealloc(p, bytes);
    }
  }

  /** Sample a continuation from the current context without learning. */
  generate(len: number, seed = 1): number[] {
    this.check();
    if (len <= 0) return [];
    const bytes = len * 4;
    const p = this.x.sp_alloc(bytes);
    try {
      const n = this.x.sp_generate(this.handle, len, seed, p);
      return [...new Uint32Array(this.x.memory.buffer, p, n)];
    } finally {
      this.x.sp_dealloc(p, bytes);
    }
  }

  /** Forget the running context but keep what was learned. */
  reset(): void {
    this.check();
    this.x.sp_reset(this.handle);
  }

  free(): void {
    if (this.handle !== 0) this.x.sp_free(this.handle);
    this.handle = 0;
  }

  private check(): void {
    if (this.handle === 0) throw new Error("engine already freed");
  }
}
