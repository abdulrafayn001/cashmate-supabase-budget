// _shared/mcmc/random.ts
// ============================================================================
// Seeded PRNG + sampling primitives for the MCMC suite.
//
// Math.random is unseedable in Deno per spec, so we ship cyrb128 (a 128-bit
// hash for seed expansion) + sfc32 (period >= 2^121, passes PractRand to 32 TB).
// The pair is the bryc-recommended combo for production MCMC: fast, well-mixed,
// and reproducible from a string seed like `${user_id}:${run_id}`.
//
// Higher-level samplers (gamma, beta, Dirichlet, MVN) take an Rng and never
// touch Math.random themselves — every draw is bit-for-bit reproducible given
// the same seed.
// ============================================================================

import type { Rng } from './types.ts'

// ---------------------------------------------------------------------------
// Seed: cyrb128 (bryc, public domain)
// Returns four 32-bit unsigned integers for sfc32 to consume.
// ---------------------------------------------------------------------------
export function cyrb128(str: string): [number, number, number, number] {
  let h1 = 1779033703
  let h2 = 3144134277
  let h3 = 1013904242
  let h4 = 2773480762
  for (let i = 0, k: number; i < str.length; i++) {
    k = str.charCodeAt(i)
    h1 = h2 ^ Math.imul(h1 ^ k, 597399067)
    h2 = h3 ^ Math.imul(h2 ^ k, 2869860233)
    h3 = h4 ^ Math.imul(h3 ^ k, 951274213)
    h4 = h1 ^ Math.imul(h4 ^ k, 2716044179)
  }
  h1 = Math.imul(h3 ^ (h1 >>> 18), 597399067)
  h2 = Math.imul(h4 ^ (h2 >>> 22), 2869860233)
  h3 = Math.imul(h1 ^ (h3 >>> 17), 951274213)
  h4 = Math.imul(h2 ^ (h4 >>> 19), 2716044179)
  h1 ^= h2 ^ h3 ^ h4
  h2 ^= h1
  h3 ^= h1
  h4 ^= h1
  return [h1 >>> 0, h2 >>> 0, h3 >>> 0, h4 >>> 0]
}

// ---------------------------------------------------------------------------
// Core PRNG: sfc32 (Chris Doty-Humphrey, public domain)
// 128-bit state; returns a fresh Rng closure that yields floats in [0, 1).
// ---------------------------------------------------------------------------
export function sfc32(a: number, b: number, c: number, d: number): Rng {
  return function (): number {
    a |= 0
    b |= 0
    c |= 0
    d |= 0
    const t = (((a + b) | 0) + d) | 0
    d = (d + 1) | 0
    a = b ^ (b >>> 9)
    b = (c + (c << 3)) | 0
    c = (c << 21) | (c >>> 11)
    c = (c + t) | 0
    return (t >>> 0) / 4294967296
  }
}

/** Build a seeded Rng from any string. Use `${user_id}:${run_id}` in prod. */
export function makeRng(seed: string): Rng {
  const s = cyrb128(seed)
  return sfc32(s[0], s[1], s[2], s[3])
}

// ---------------------------------------------------------------------------
// Standard normal: Box-Muller (polar form), with cached spare.
// We carry the spare per-Rng via a WeakMap so multiple chains stay independent
// without re-rolling spares from each other's streams.
// ---------------------------------------------------------------------------
const spareMap = new WeakMap<Rng, number | null>()

export function stdNormal(rng: Rng): number {
  const spare = spareMap.get(rng)
  if (spare !== undefined && spare !== null) {
    spareMap.set(rng, null)
    return spare
  }
  // Polar Box-Muller — rejection-based, no trig calls.
  let u: number, v: number, s: number
  do {
    u = 2 * rng() - 1
    v = 2 * rng() - 1
    s = u * u + v * v
  } while (s === 0 || s >= 1)
  const mul = Math.sqrt((-2 * Math.log(s)) / s)
  spareMap.set(rng, v * mul)
  return u * mul
}

// ---------------------------------------------------------------------------
// Gamma(shape, scale): Marsaglia & Tsang 2000 + Marsaglia boost for shape<1.
// Returns a draw with mean = shape * scale, variance = shape * scale^2.
// ---------------------------------------------------------------------------
export function gamma(rng: Rng, shape: number, scale = 1): number {
  if (shape < 1) {
    // Boost shape: G(α) = G(α + 1) * U^(1/α)
    const u = rng()
    return gamma(rng, shape + 1, scale) * Math.pow(u, 1 / shape)
  }
  const d = shape - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  while (true) {
    let x: number, v: number
    do {
      x = stdNormal(rng)
      v = 1 + c * x
    } while (v <= 0)
    v = v * v * v
    const u = rng()
    if (u < 1 - 0.0331 * x * x * x * x) return d * v * scale
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
      return d * v * scale
    }
  }
}

/** Beta(α, β) from two Gammas. Mean = α / (α + β). */
export function beta(rng: Rng, alpha: number, betaParam: number): number {
  const x = gamma(rng, alpha, 1)
  const y = gamma(rng, betaParam, 1)
  return x / (x + y)
}

/** Dirichlet(α[1..K]) → simplex of length K. */
export function dirichlet(rng: Rng, alpha: Float64Array | number[]): Float64Array {
  const K = alpha.length
  const out = new Float64Array(K)
  let sum = 0
  for (let i = 0; i < K; i++) {
    out[i] = gamma(rng, alpha[i], 1)
    sum += out[i]
  }
  if (sum > 0) {
    for (let i = 0; i < K; i++) out[i] /= sum
  }
  return out
}

/** Inverse-Gamma(α, β): if X ~ Gamma(α, 1/β), then 1/X ~ IG(α, β). */
export function inverseGamma(rng: Rng, alpha: number, betaParam: number): number {
  return 1 / gamma(rng, alpha, 1 / betaParam)
}

/** Half-Cauchy(σ): |X| where X ~ Cauchy(0, σ). Sampled via inverse CDF. */
export function halfCauchy(rng: Rng, sigma: number): number {
  // Inverse CDF: F^{-1}(p) = sigma * tan(π/2 * p) for p in [0, 1)
  return sigma * Math.tan((Math.PI / 2) * rng())
}

// ---------------------------------------------------------------------------
// Multivariate normal: μ + L · z where L = Cholesky(Σ), z ~ N(0, I_d).
// `mu` and `L` are taken row-major; `out` is filled in place if provided.
// ---------------------------------------------------------------------------
export function mvn(
  rng: Rng,
  mu: Float64Array,
  L: Float64Array,
  d: number,
  out?: Float64Array,
): Float64Array {
  const result = out ?? new Float64Array(d)
  const z = new Float64Array(d)
  for (let i = 0; i < d; i++) z[i] = stdNormal(rng)
  for (let i = 0; i < d; i++) {
    let s = 0
    // L is lower-triangular, so j ranges 0..i.
    for (let j = 0; j <= i; j++) s += L[i * d + j] * z[j]
    result[i] = mu[i] + s
  }
  return result
}
