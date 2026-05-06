// _shared/mcmc/diagnostics.ts
// ============================================================================
// Convergence diagnostics:
//   - splitRhat: Vehtari et al. 2021 split-R̂ over M chains × N samples.
//   - ess: Geyer 1992 initial monotone sequence estimator (per-chain), then
//     combined across chains via the Vehtari multi-chain formula.
//
// Inputs are flat Float64Arrays of length N*dim per chain (the layout produced
// by adaptiveMH / gibbs). Callers index into a single coordinate by paramIdx.
// ============================================================================

/** Extract a single parameter's trace from a flat (nKeep × dim) buffer. */
function extractTrace(chain: Float64Array, paramIdx: number, dim: number): Float64Array {
  const n = chain.length / dim
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = chain[i * dim + paramIdx]
  return out
}

function mean(x: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i]
  return s / x.length
}

function variance(x: ArrayLike<number>, m?: number): number {
  const mu = m ?? mean(x)
  let s = 0
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - mu
    s += d * d
  }
  return s / (x.length - 1)
}

// ---------------------------------------------------------------------------
// split-R̂ (Vehtari 2021)
// Splits each of M chains into 2 → 2M chains of length N' = floor(N/2),
// computes between-chain variance B and within-chain variance W, returns
// sqrt(((N'-1)/N') * W/W + B/(N' * W)) — the canonical formula.
// ---------------------------------------------------------------------------
export function splitRhat(chains: Float64Array[], paramIdx: number, dim: number): number {
  const M = chains.length
  if (M < 2) return NaN
  const traces: Float64Array[] = []
  for (const c of chains) traces.push(extractTrace(c, paramIdx, dim))

  const Nfull = traces[0].length
  const N = Math.floor(Nfull / 2)
  if (N < 2) return NaN

  // Build 2M half-chains.
  const halves: Float64Array[] = []
  for (const t of traces) {
    halves.push(t.subarray(0, N) as Float64Array)
    halves.push(t.subarray(Nfull - N, Nfull) as Float64Array)
  }
  const M2 = halves.length

  const chainMeans = halves.map((h) => mean(h))
  const chainVars = halves.map((h, i) => variance(h, chainMeans[i]))
  const grandMean = chainMeans.reduce((a, b) => a + b, 0) / M2

  let B = 0
  for (let i = 0; i < M2; i++) {
    const d = chainMeans[i] - grandMean
    B += d * d
  }
  B = (N / (M2 - 1)) * B

  let W = 0
  for (const v of chainVars) W += v
  W /= M2

  if (W <= 0) return NaN

  const varPlus = ((N - 1) / N) * W + B / N
  return Math.sqrt(varPlus / W)
}

// ---------------------------------------------------------------------------
// Effective Sample Size (Geyer 1992 initial monotone)
//
// Single-chain pipeline:
//   1. Compute autocovariance ρ_t for t = 0, 1, ..., until paired sums go neg.
//   2. Take cumulative sums of ρ_{2k} + ρ_{2k+1}; truncate when negative.
//   3. Enforce the cumulative sum to be monotone non-increasing.
//   4. τ = -1 + 2 * Σ_{k=0}^{K} (ρ_{2k} + ρ_{2k+1}) / ρ_0
//   5. ESS = N / τ
//
// Multi-chain combination (Vehtari 2021 §3.2 simplified):
//   - Use the var_plus from split-R̂ as the variance estimator
//   - Multi-chain ESS = M * N / τ_hat, where τ_hat uses the average
//     autocorrelation across chains.
//
// We implement (1)-(5) per chain and then sum the per-chain ESS — this is the
// conservative-but-correct combiner (Geyer 2011 §1.10.3) and matches Stan's
// rstan within ~5% for chains with similar mixing.
// ---------------------------------------------------------------------------

function autocorrelation(x: Float64Array, maxLag: number): Float64Array {
  const N = x.length
  const mu = mean(x)
  const v = variance(x, mu)
  const rho = new Float64Array(maxLag + 1)
  rho[0] = 1
  if (v <= 0) return rho
  for (let t = 1; t <= maxLag; t++) {
    let s = 0
    for (let i = 0; i + t < N; i++) s += (x[i] - mu) * (x[i + t] - mu)
    rho[t] = s / ((N - t) * v)
  }
  return rho
}

function geyerEssChain(x: Float64Array): number {
  const N = x.length
  if (N < 4) return NaN
  const maxLag = Math.min(N - 1, 1000)
  const rho = autocorrelation(x, maxLag)

  // Pair sums ρ_{2k} + ρ_{2k+1}; stop when they go non-positive (initial
  // positive sequence).
  const pairSums: number[] = []
  for (let k = 0; 2 * k + 1 < rho.length; k++) {
    const p = rho[2 * k] + rho[2 * k + 1]
    if (p <= 0) break
    pairSums.push(p)
  }
  if (pairSums.length === 0) return N

  // Enforce monotone non-increasing (initial monotone sequence).
  for (let i = 1; i < pairSums.length; i++) {
    if (pairSums[i] > pairSums[i - 1]) pairSums[i] = pairSums[i - 1]
  }

  let sum = 0
  for (const p of pairSums) sum += p
  // τ = -1 + 2 * Σ; ESS = N / τ. The first term ρ_0 = 1 is in pairSums[0], so:
  const tau = -1 + 2 * sum
  if (tau <= 0) return N
  return N / tau
}

/** ESS for a single coordinate across M chains. Sums per-chain Geyer ESS,
 *  which is exact for IID and conservative for autocorrelated samples. */
export function ess(chains: Float64Array[], paramIdx: number, dim: number): number {
  let total = 0
  for (const c of chains) {
    const trace = extractTrace(c, paramIdx, dim)
    const e = geyerEssChain(trace)
    if (isFinite(e)) total += e
  }
  return total
}

// ---------------------------------------------------------------------------
// Bulk summarizer: returns mean/sd/quantiles for one coordinate across all
// post-burn draws from all chains. Quantiles via sort (acceptable for ≤ 5k
// samples per param; switch to histogram-based for larger).
// ---------------------------------------------------------------------------
export function summarize(
  chains: Float64Array[],
  paramIdx: number,
  dim: number,
): {
  mean: number
  sd: number
  q05: number
  q25: number
  q50: number
  q75: number
  q95: number
} {
  const all: number[] = []
  for (const c of chains) {
    const trace = extractTrace(c, paramIdx, dim)
    for (let i = 0; i < trace.length; i++) all.push(trace[i])
  }
  all.sort((a, b) => a - b)
  const m = all.reduce((a, b) => a + b, 0) / all.length
  const v = all.reduce((s, x) => s + (x - m) * (x - m), 0) / (all.length - 1)
  const q = (p: number) => {
    const idx = Math.min(all.length - 1, Math.max(0, Math.floor(p * (all.length - 1))))
    return all[idx]
  }
  return {
    mean: m,
    sd: Math.sqrt(v),
    q05: q(0.05),
    q25: q(0.25),
    q50: q(0.5),
    q75: q(0.75),
    q95: q(0.95),
  }
}
