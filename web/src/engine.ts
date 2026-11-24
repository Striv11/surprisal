// Thin typed wrapper over the hand-written C ABI in src/wasm.rs.

interface Exports {
  memory: WebAssembly.Memory;
  sp_create(vocab: number, hidden: number, seed: number): number;
  sp_free(h: number): void;
  sp_observe(h: number, symbol: number): number;
  sp_predict(h: number, out: number, cap: number): number;
  sp_generate(h: number, len: number, seed: number, out: number): number;
  sp_reset(h: number): void;
  sp_alloc(bytes: number): number;
  sp_dealloc(p: number, bytes: number): void;
}

export class Engine {
  private handle: number;
  private x: Exports;
  readonly vocab: number;
  readonly hidden: number;
  readonly seed: number;

  private constructor(x: Exports, vocab: number, hidden: number, seed: number) {
    this.x = x;
    this.vocab = vocab;
    this.hidden = hidden;
    this.seed = seed;
    this.handle = x.sp_create(vocab, hidden, seed);
    if (this.handle === 0) throw new Error("sp_create rejected the configuration");
  }

  static async load(
    wasm: BufferSource | Response | Promise<Response>,
    vocab = 7,
    hidden = 24,
    seed = 42,
  ): Promise<Engine> {
    let instance: WebAssembly.Instance;
    if (wasm instanceof Response || wasm instanceof Promise) {
      ({ instance } = await WebAssembly.instantiateStreaming(wasm, {}));
    } else {
      ({ instance } = await WebAssembly.instantiate(wasm, {}));
    }
    return new Engine(instance.exports as unknown as Exports, vocab, hidden, seed);
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
