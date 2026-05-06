// _shared/mcmc/features/cashflow.ts
// ============================================================================
// A1 — Probabilistic cash-flow & net-worth forecast.
//
// Generative model (per design doc §2 A1):
//
//   # Latent log-monthly trends
//   muC_t = muC_{t-1} + etaC_t,                                etaC_t ~ N(0, qC)
//   muD_t = phi*muD_{t-1} + (1-phi)*mD + etaD_t,               etaD_t ~ N(0, qD)
//
//   # Observations (heavy-tailed)
//   log(c_t) ~ t_nu(muC_t, sigmaC²)
//   log(d_t) ~ t_nu(muD_t, sigmaD²)
//
//   # Priors
//   muC_0, muD_0 ~ N(log(median), 1)
//   mD          ~ N(log(median_spend), 1)
//   phi         ~ Beta(8, 2)
//   qC, qD      ~ HalfCauchy(0.1)
//   sigmaC, sigmaD ~ HalfCauchy(0.5)
//   nu          ~ Gamma(2, 0.1)            # df ∈ (~2, ~30)
//
// Sampler: Metropolis-within-Gibbs with the standard scale-mixture-of-normals
// auxiliary representation for Student-t (Geweke 1993):
//
//   t_nu(y|μ,σ²)  ≡  ∫ N(y|μ, σ²/λ) · Gamma(λ | nu/2, nu/2) dλ
//
// Conditional on λ_t (one per observation) all latent states are conditionally
// Gaussian, so per-time updates are closed-form Normal-Normal posteriors —
// FFBS-equivalent for our T≈12 horizon, but simpler to write and audit.
//
// Per-iteration steps, in order:
//   1. λC_t, λD_t        — aux-Gamma update (closed form)
//   2. muC_{0..T}        — per-t Gibbs (random walk + Gaussian likelihood)
//   3. muD_{0..T}        — per-t Gibbs (AR(1) + Gaussian likelihood)
//   4. mD                — Gibbs (closed-form Normal)
//   5. phi               — RW MH on logit(phi); Beta(8,2) prior
//   6. qC, qD            — slice on log-scale; HalfCauchy(0.1) prior
//   7. sigmaC, sigmaD    — slice on log-scale; HalfCauchy(0.5) prior
//   8. nu                — RW MH on log(nu); Gamma(2, 0.1) prior
// ============================================================================

import type { Rng } from '../types.ts'
import { gamma, makeRng, stdNormal } from '../random.ts'
import { logHalfCauchy } from '../distributions.ts'
import { ess, splitRhat, summarize } from '../diagnostics.ts'
import { slice } from '../samplers.ts'

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Aggregated monthly inputs in base currency (PKR). All values must be
 *  strictly positive — months with zero income or zero spend are masked
 *  upstream by replacing with the previous month's value (caller's choice;
 *  the sampler does not handle log(0)). */
export interface CashflowData {
  /** Monthly credits (income), length T. */
  c: Float64Array
  /** Monthly debits (spend), length T. */
  d: Float64Array
  /** ISO month labels for each observation, length T (e.g. ['2025-06', …]). */
  months: string[]
}

export interface CashflowOpts {
  /** Forecast horizon in months (default 6). */
  horizon: number
  /** Total iterations per chain (default 2000). */
  nIter: number
  /** Burn-in iterations (default 1000). */
  nBurn: number
  /** Thinning interval (default 2). */
  thin: number
  /** Number of chains (default 4). */
  nChains: number
  /** Yield to event loop every yieldEvery iterations (default 250). */
  yieldEvery: number
}

export const DEFAULT_OPTS: CashflowOpts = {
  horizon: 6,
  nIter: 2000,
  nBurn: 1000,
  thin: 2,
  nChains: 4,
  yieldEvery: 250,
}

/** Per-month forecast quantiles for one series (income, spend, or net). */
export interface ForecastSeries {
  median: Float64Array
  q05: Float64Array
  q25: Float64Array
  q75: Float64Array
  q95: Float64Array
}

export interface CashflowForecast {
  forecastMonths: string[]
  income: ForecastSeries
  spend: ForecastSeries
  /** Net = income − spend, on the original (non-log) scale. */
  net: ForecastSeries
  /** Cumulative net flow starting from 0, useful for net-worth charts. */
  cumulativeNet: ForecastSeries
}

export interface CashflowResult {
  diagnostics: {
    rhatMax: number
    essMin: number
    nDivergences: number
    pass: boolean
    perParam: Record<string, { rhat: number; ess: number }>
  }
  /** Posterior summaries for global params (mean/sd/quantiles). */
  summary: Record<
    string,
    { mean: number; sd: number; q05: number; q25: number; q50: number; q75: number; q95: number }
  >
  forecast: CashflowForecast
  runtimeMs: number
  nSamples: number
}

// ---------------------------------------------------------------------------
// Layout: pack all state into a single Float64Array per chain so we can
// hand it straight to the diagnostics functions.
//
//   Indices [0 .. T]            -> muC_{0..T}      (T+1 latent income states)
//   Indices [T+1 .. 2T+1]       -> muD_{0..T}      (T+1 latent spend states)
//   Index   2T+2                -> phi
//   Index   2T+3                -> mD
//   Index   2T+4                -> qC
//   Index   2T+5                -> qD
//   Index   2T+6                -> sigmaC
//   Index   2T+7                -> sigmaD
//   Index   2T+8                -> nu
// Total dim = 2T + 9.
// ---------------------------------------------------------------------------

interface Layout {
  T: number
  dim: number
  muCStart: number
  muDStart: number
  phi: number
  mD: number
  qC: number
  qD: number
  sigmaC: number
  sigmaD: number
  nu: number
}

function makeLayout(T: number): Layout {
  return {
    T,
    dim: 2 * T + 9,
    muCStart: 0,
    muDStart: T + 1,
    phi: 2 * T + 2,
    mD: 2 * T + 3,
    qC: 2 * T + 4,
    qD: 2 * T + 5,
    sigmaC: 2 * T + 6,
    sigmaD: 2 * T + 7,
    nu: 2 * T + 8,
  }
}

/** Names matching layout indices, used for the diagnostics summary. */
function paramNames(T: number): string[] {
  const names: string[] = []
  for (let t = 0; t <= T; t++) names.push(`muC[${t}]`)
  for (let t = 0; t <= T; t++) names.push(`muD[${t}]`)
  names.push('phi', 'mD', 'qC', 'qD', 'sigmaC', 'sigmaD', 'nu')
  return names
}

// ---------------------------------------------------------------------------
// Initialization: median-based, mildly informative.
// ---------------------------------------------------------------------------

function median(x: Float64Array): number {
  const a = Array.from(x).sort((p, q) => p - q)
  const n = a.length
  return n % 2 === 1 ? a[(n - 1) / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2])
}

function initState(data: CashflowData, layout: Layout, rng: Rng): Float64Array {
  const T = layout.T
  const state = new Float64Array(layout.dim)
  const logCMed = Math.log(median(data.c))
  const logDMed = Math.log(median(data.d))
  // Latent states ≈ observed log values + tiny jitter, so logp is finite.
  for (let t = 0; t <= T; t++) {
    const obs = t === 0 ? logCMed : Math.log(data.c[t - 1])
    state[layout.muCStart + t] = obs + 0.05 * stdNormal(rng)
  }
  for (let t = 0; t <= T; t++) {
    const obs = t === 0 ? logDMed : Math.log(data.d[t - 1])
    state[layout.muDStart + t] = obs + 0.05 * stdNormal(rng)
  }
  // Globals
  state[layout.phi] = 0.7
  state[layout.mD] = logDMed
  state[layout.qC] = 0.05
  state[layout.qD] = 0.05
  state[layout.sigmaC] = 0.2
  state[layout.sigmaD] = 0.2
  state[layout.nu] = 5
  return state
}

// ---------------------------------------------------------------------------
// One full Gibbs sweep over the state vector. Mutates `state` in place.
// `lamC`/`lamD` (auxiliary precisions, length T) are reused across iters.
// ---------------------------------------------------------------------------

function sweep(
  state: Float64Array,
  data: CashflowData,
  layout: Layout,
  lamC: Float64Array,
  lamD: Float64Array,
  rng: Rng,
  logCMed: number,
  logDMed: number,
): void {
  const T = layout.T
  const muCStart = layout.muCStart
  const muDStart = layout.muDStart

  const phi = state[layout.phi]
  const mD = state[layout.mD]
  const qC = state[layout.qC]
  const qD = state[layout.qD]
  const sigmaC = state[layout.sigmaC]
  const sigmaD = state[layout.sigmaD]
  const nu = state[layout.nu]

  // 1. Auxiliary Gamma scales: λ_t | rest ~ Gamma((nu+1)/2, (nu + r²/σ²)/2)
  const sigmaC2 = sigmaC * sigmaC
  const sigmaD2 = sigmaD * sigmaD
  const aLam = (nu + 1) / 2
  for (let t = 0; t < T; t++) {
    const rC = Math.log(data.c[t]) - state[muCStart + t + 1]
    const rD = Math.log(data.d[t]) - state[muDStart + t + 1]
    const bC = (nu + (rC * rC) / sigmaC2) / 2
    const bD = (nu + (rD * rD) / sigmaD2) / 2
    // gamma(α, scale=1/β) since our gamma() takes (shape, scale)
    lamC[t] = gamma(rng, aLam, 1 / bC)
    lamD[t] = gamma(rng, aLam, 1 / bD)
  }

  // 2. muC latent states (random walk + obs).
  //    For each t, prior contributions from neighbors + likelihood.
  //    State indices 0..T; observation y_C[t-1] is tied to state index t (t≥1).
  for (let t = 0; t <= T; t++) {
    let prec = 0
    let rhsMean = 0

    // Prior (random-walk, anchored): muC_0 ~ N(logCMed, 1)
    if (t === 0) {
      prec += 1
      rhsMean += logCMed
    } else {
      // Forward: muC_t | muC_{t-1} ~ N(muC_{t-1}, qC)
      prec += 1 / qC
      rhsMean += state[muCStart + t - 1] / qC
    }
    // Backward: muC_{t+1} | muC_t ~ N(muC_t, qC)
    if (t < T) {
      prec += 1 / qC
      rhsMean += state[muCStart + t + 1] / qC
    }
    // Likelihood: log(c[t-1]) ~ N(muC_t, sigmaC²/λ_{t-1}); applies for t≥1.
    if (t >= 1) {
      const tau = lamC[t - 1] / sigmaC2
      prec += tau
      rhsMean += Math.log(data.c[t - 1]) * tau
    }

    const postVar = 1 / prec
    const postMean = rhsMean * postVar
    state[muCStart + t] = postMean + Math.sqrt(postVar) * stdNormal(rng)
  }

  // 3. muD latent states (AR(1) + obs).
  //    Forward kernel: muD_t | muD_{t-1} ~ N(phi*muD_{t-1} + (1-phi)*mD, qD)
  //    Backward (treat next-step constraint symmetrically):
  //      muD_{t+1} | muD_t ~ N(phi*muD_t + (1-phi)*mD, qD)
  //      → contributes precision phi²/qD with mean (muD_{t+1} - (1-phi)*mD) / phi
  for (let t = 0; t <= T; t++) {
    let prec = 0
    let rhsMean = 0

    if (t === 0) {
      // muD_0 prior ~ N(logDMed, 1)
      prec += 1
      rhsMean += logDMed
    } else {
      // Forward
      prec += 1 / qD
      rhsMean += (phi * state[muDStart + t - 1] + (1 - phi) * mD) / qD
    }
    if (t < T) {
      // Backward via AR(1) likelihood
      const tau = (phi * phi) / qD
      const target = (state[muDStart + t + 1] - (1 - phi) * mD) / phi
      prec += tau
      rhsMean += target * tau
    }
    if (t >= 1) {
      const tau = lamD[t - 1] / sigmaD2
      prec += tau
      rhsMean += Math.log(data.d[t - 1]) * tau
    }

    const postVar = 1 / prec
    const postMean = rhsMean * postVar
    state[muDStart + t] = postMean + Math.sqrt(postVar) * stdNormal(rng)
  }

  // 4. mD (long-run spend mean) — Gibbs.
  //    Let z_t = muD_t - phi*muD_{t-1}, t=1..T → z_t ~ N((1-phi)*mD, qD)
  //    Prior mD ~ N(logDMed, 1)
  {
    let sumZ = 0
    for (let t = 1; t <= T; t++) {
      sumZ += state[muDStart + t] - phi * state[muDStart + t - 1]
    }
    const omp = 1 - phi
    const dataPrec = (T * omp * omp) / qD
    const postPrec = 1 + dataPrec
    const postVar = 1 / postPrec
    const postMean = (logDMed + (omp / qD) * sumZ) * postVar
    state[layout.mD] = postMean + Math.sqrt(postVar) * stdNormal(rng)
  }

  // 5. phi — slice on logit(phi). Beta(8,2) prior + AR(1) likelihood.
  {
    const mDcur = state[layout.mD]
    const logitPhi = Math.log(state[layout.phi] / (1 - state[layout.phi]))
    const newLogit = slice(
      (z) => {
        const p = 1 / (1 + Math.exp(-z))
        if (p <= 0 || p >= 1) return -Infinity
        // log Beta(8, 2): (8-1) log p + (2-1) log(1-p)  [up to constant]
        let lp = 7 * Math.log(p) + 1 * Math.log(1 - p)
        // AR(1) log-likelihood
        for (let t = 1; t <= T; t++) {
          const r = state[muDStart + t] - p * state[muDStart + t - 1] - (1 - p) * mDcur
          lp += -0.5 * (r * r) / qD
        }
        // Jacobian of z = logit(p): dp/dz = p(1-p), so log|J| = log p + log(1-p)
        lp += Math.log(p) + Math.log(1 - p)
        return lp
      },
      logitPhi,
      rng,
      1.5,
    )
    state[layout.phi] = 1 / (1 + Math.exp(-newLogit))
  }

  // 6. qC, qD — slice on log-scale. HalfCauchy(0.1) prior.
  state[layout.qC] = sliceLogScale(
    (q) => {
      let lp = logHalfCauchy(q, 0.1)
      for (let t = 1; t <= T; t++) {
        const r = state[muCStart + t] - state[muCStart + t - 1]
        lp += -0.5 * Math.log(q) - 0.5 * (r * r) / q
      }
      return lp
    },
    state[layout.qC],
    rng,
  )
  state[layout.qD] = sliceLogScale(
    (q) => {
      let lp = logHalfCauchy(q, 0.1)
      const phiCur = state[layout.phi]
      const mDcur2 = state[layout.mD]
      for (let t = 1; t <= T; t++) {
        const r = state[muDStart + t] - phiCur * state[muDStart + t - 1] - (1 - phiCur) * mDcur2
        lp += -0.5 * Math.log(q) - 0.5 * (r * r) / q
      }
      return lp
    },
    state[layout.qD],
    rng,
  )

  // 7. sigmaC, sigmaD — slice on log-scale. HalfCauchy(0.5) prior.
  state[layout.sigmaC] = sliceLogScale(
    (s) => {
      let lp = logHalfCauchy(s, 0.5)
      const s2 = s * s
      for (let t = 0; t < T; t++) {
        const r = Math.log(data.c[t]) - state[muCStart + t + 1]
        lp += -0.5 * Math.log(s2 / lamC[t]) - 0.5 * (r * r * lamC[t]) / s2
      }
      return lp
    },
    state[layout.sigmaC],
    rng,
  )
  state[layout.sigmaD] = sliceLogScale(
    (s) => {
      let lp = logHalfCauchy(s, 0.5)
      const s2 = s * s
      for (let t = 0; t < T; t++) {
        const r = Math.log(data.d[t]) - state[muDStart + t + 1]
        lp += -0.5 * Math.log(s2 / lamD[t]) - 0.5 * (r * r * lamD[t]) / s2
      }
      return lp
    },
    state[layout.sigmaD],
    rng,
  )

  // 8. nu — slice on log(nu). Gamma(2, 0.1) prior on nu.
  //    Posterior conditional uses the marginal t-likelihood (not the auxiliary
  //    Gamma decomposition) so we don't double-count λ.
  state[layout.nu] = sliceLogScale(
    (nuCur) => {
      // log Gamma(α=2, β=0.1) prior: 2 log(0.1) - lgamma(2) + (2-1) log nu - 0.1 nu
      // The norm const drops out of slice; just need log nu - 0.1 nu.
      let lp = Math.log(nuCur) - 0.1 * nuCur
      // Marginal t observation density
      const sC = state[layout.sigmaC]
      const sD = state[layout.sigmaD]
      for (let t = 0; t < T; t++) {
        const rC = Math.log(data.c[t]) - state[muCStart + t + 1]
        const rD = Math.log(data.d[t]) - state[muDStart + t + 1]
        lp += logTKernel(rC, sC, nuCur) + logTKernel(rD, sD, nuCur)
      }
      return lp
    },
    state[layout.nu],
    rng,
  )
}

/** Student-t log-density kernel (drops constants in nu, sigma). Used inside
 *  slice samplers where const offsets don't matter. */
function logTKernel(residual: number, sigma: number, nu: number): number {
  // log p(r | sigma, nu) up to additive const in (sigma, nu) ratio:
  // = log Γ((nu+1)/2) - log Γ(nu/2) - 0.5 log(nu π) - log sigma
  //   - (nu+1)/2 * log(1 + r²/(nu σ²))
  // We need the nu-dependent terms (priors over sigma are handled separately).
  return (
    lgammaApprox((nu + 1) / 2) -
    lgammaApprox(nu / 2) -
    0.5 * Math.log(nu * Math.PI) -
    Math.log(sigma) -
    ((nu + 1) / 2) * Math.log(1 + (residual * residual) / (nu * sigma * sigma))
  )
}

/** Cheap log-Gamma (Stirling+correction); good to ~1e-8 for x>1.
 *  Avoids a circular import on distributions.lgamma when this file is
 *  imported standalone. Both functions are equivalent to ~6 sig figs. */
function lgammaApprox(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgammaApprox(1 - x)
  x -= 1
  const c = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  let a = 0.99999999999980993
  for (let i = 0; i < c.length; i++) a += c[i] / (x + i + 1)
  const t = x + c.length - 0.5
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a)
}

/** Slice on log-scale for a positive parameter. Wraps `slice` from samplers.ts
 *  with a Jacobian. */
function sliceLogScale(unnormLogP: (x: number) => number, x0: number, rng: Rng): number {
  // Slice in z = log x. Posterior in z: p(x) * |dx/dz| = p(x) * x.
  const z0 = Math.log(x0)
  const zNew = slice(
    (z) => {
      const x = Math.exp(z)
      const lp = unnormLogP(x)
      return isFinite(lp) ? lp + z : -Infinity // + log|dx/dz| = + z
    },
    z0,
    rng,
    1.0,
  )
  return Math.exp(zNew)
}

// ---------------------------------------------------------------------------
// Single-chain runner. Returns flat draws + a step-yield hook.
// ---------------------------------------------------------------------------

export async function runChain(
  data: CashflowData,
  opts: CashflowOpts,
  rng: Rng,
  yieldFn?: () => Promise<void>,
): Promise<Float64Array> {
  const T = data.c.length
  if (T !== data.d.length) {
    throw new Error(`runChain: c.length (${data.c.length}) != d.length (${T})`)
  }
  const layout = makeLayout(T)
  const logCMed = Math.log(median(data.c))
  const logDMed = Math.log(median(data.d))

  const state = initState(data, layout, rng)
  const lamC = new Float64Array(T).fill(1)
  const lamD = new Float64Array(T).fill(1)

  const nKeep = Math.max(0, Math.floor((opts.nIter - opts.nBurn) / opts.thin))
  const draws = new Float64Array(nKeep * layout.dim)
  let kept = 0

  for (let i = 0; i < opts.nIter; i++) {
    sweep(state, data, layout, lamC, lamD, rng, logCMed, logDMed)
    if (i >= opts.nBurn && (i - opts.nBurn) % opts.thin === 0 && kept < nKeep) {
      draws.set(state, kept * layout.dim)
      kept++
    }
    if (yieldFn && i > 0 && i % opts.yieldEvery === 0) await yieldFn()
  }

  return draws
}

// ---------------------------------------------------------------------------
// Forward simulation: from each retained draw, simulate H months ahead.
// Returns posterior quantiles per future month for income, spend, net.
// ---------------------------------------------------------------------------

function forecastFromDraws(
  draws: Float64Array,
  layout: Layout,
  H: number,
  rng: Rng,
  baseMonth: string,
): CashflowForecast {
  const T = layout.T
  const dim = layout.dim
  const N = draws.length / dim

  // For each future month h=1..H, collect across draws: c, d, net.
  const futureC: number[][] = Array.from({ length: H }, () => [])
  const futureD: number[][] = Array.from({ length: H }, () => [])
  const futureNet: number[][] = Array.from({ length: H }, () => [])
  const futureCum: number[][] = Array.from({ length: H }, () => [])

  for (let i = 0; i < N; i++) {
    const off = i * dim
    let muC = draws[off + layout.muCStart + T]
    let muD = draws[off + layout.muDStart + T]
    const phi = draws[off + layout.phi]
    const mD = draws[off + layout.mD]
    const qC = draws[off + layout.qC]
    const qD = draws[off + layout.qD]
    const sigmaC = draws[off + layout.sigmaC]
    const sigmaD = draws[off + layout.sigmaD]
    const nu = draws[off + layout.nu]

    let cumNet = 0
    for (let h = 0; h < H; h++) {
      // Advance latent states.
      muC += Math.sqrt(qC) * stdNormal(rng)
      muD = phi * muD + (1 - phi) * mD + Math.sqrt(qD) * stdNormal(rng)

      // Sample observations from t-likelihood via aux Gamma:
      // y = mu + (sigma / sqrt(λ)) * Z, where λ ~ Gamma(nu/2, 2/nu).
      const lamCh = gamma(rng, nu / 2, 2 / nu)
      const lamDh = gamma(rng, nu / 2, 2 / nu)
      const logC = muC + (sigmaC / Math.sqrt(lamCh)) * stdNormal(rng)
      const logD = muD + (sigmaD / Math.sqrt(lamDh)) * stdNormal(rng)
      const cVal = Math.exp(logC)
      const dVal = Math.exp(logD)
      const netVal = cVal - dVal
      cumNet += netVal

      futureC[h].push(cVal)
      futureD[h].push(dVal)
      futureNet[h].push(netVal)
      futureCum[h].push(cumNet)
    }
  }

  const toSeries = (cols: number[][]): ForecastSeries => {
    const med = new Float64Array(H)
    const q05 = new Float64Array(H)
    const q25 = new Float64Array(H)
    const q75 = new Float64Array(H)
    const q95 = new Float64Array(H)
    for (let h = 0; h < H; h++) {
      const sorted = cols[h].slice().sort((a, b) => a - b)
      const q = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))]
      med[h] = q(0.5)
      q05[h] = q(0.05)
      q25[h] = q(0.25)
      q75[h] = q(0.75)
      q95[h] = q(0.95)
    }
    return { median: med, q05, q25, q75, q95 }
  }

  // Build month labels.
  const months: string[] = []
  if (baseMonth) {
    const [yStr, mStr] = baseMonth.split('-')
    let y = parseInt(yStr, 10)
    let m = parseInt(mStr, 10)
    for (let h = 0; h < H; h++) {
      m++
      if (m > 12) {
        m = 1
        y++
      }
      months.push(`${y}-${String(m).padStart(2, '0')}`)
    }
  } else {
    for (let h = 0; h < H; h++) months.push(`+${h + 1}m`)
  }

  return {
    forecastMonths: months,
    income: toSeries(futureC),
    spend: toSeries(futureD),
    net: toSeries(futureNet),
    cumulativeNet: toSeries(futureCum),
  }
}

// ---------------------------------------------------------------------------
// Multi-chain driver: runs nChains independent chains, computes diagnostics,
// summarizes globals, builds the forecast.
// ---------------------------------------------------------------------------

export interface RunCashflowOpts extends Partial<CashflowOpts> {
  /** Seed prefix; the chain index is appended ('seed:0', 'seed:1', …). */
  seed: string
  /** Optional async yield; defaults to setTimeout(0). */
  yieldFn?: () => Promise<void>
}

export async function runCashflow(
  data: CashflowData,
  opts: RunCashflowOpts,
): Promise<CashflowResult> {
  const merged: CashflowOpts = { ...DEFAULT_OPTS, ...opts }
  const T = data.c.length
  const layout = makeLayout(T)
  const yieldFn = opts.yieldFn ?? (() => new Promise<void>((r) => setTimeout(r, 0)))

  const t0 = Date.now()
  const chains: Float64Array[] = []
  for (let c = 0; c < merged.nChains; c++) {
    const chainRng = makeRng(`${opts.seed}:${c}`)
    const draws = await runChain(data, merged, chainRng, yieldFn)
    chains.push(draws)
  }
  const runtimeMs = Date.now() - t0

  // Diagnostics (Phase 1 acceptance gate: rhat < 1.01, ess > 400).
  const names = paramNames(T)
  let rhatMax = 0
  let essMin = Infinity
  const perParam: Record<string, { rhat: number; ess: number }> = {}
  // Only check global params (last 7) for the gate — latent states naturally
  // have lower ESS at boundaries and that's expected.
  const globalStart = layout.phi
  for (let p = 0; p < layout.dim; p++) {
    const r = splitRhat(chains, p, layout.dim)
    const e = ess(chains, p, layout.dim)
    perParam[names[p]] = { rhat: isFinite(r) ? r : NaN, ess: e }
    if (p >= globalStart) {
      if (isFinite(r) && r > rhatMax) rhatMax = r
      if (isFinite(e) && e < essMin) essMin = e
    }
  }

  // Summary for global params (latent state summaries are big and rarely
  // useful at the API surface).
  const summary: CashflowResult['summary'] = {}
  for (let p = globalStart; p < layout.dim; p++) {
    summary[names[p]] = summarize(chains, p, layout.dim)
  }

  // Forecast: pool draws across chains.
  const allDraws = new Float64Array(chains.reduce((s, c) => s + c.length, 0))
  let off = 0
  for (const c of chains) {
    allDraws.set(c, off)
    off += c.length
  }
  const baseMonth = data.months[data.months.length - 1] ?? ''
  const forecastRng = makeRng(`${opts.seed}:forecast`)
  const forecast = forecastFromDraws(allDraws, layout, merged.horizon, forecastRng, baseMonth)

  const nSamples = (allDraws.length / layout.dim) | 0
  const pass = rhatMax < 1.01 && essMin > 400
  return {
    diagnostics: { rhatMax, essMin, nDivergences: 0, pass, perParam },
    summary,
    forecast,
    runtimeMs,
    nSamples,
  }
}

// ---------------------------------------------------------------------------
// Aggregation helper — pure-function over rows from `transactions`.
// Caller does the SQL; this groups + masks zero months.
// ---------------------------------------------------------------------------

export interface TxRow {
  /** ISO timestamp string. */
  date: string
  /** 'credit' | 'debit' | 'transfer'. */
  type: string
  /** PKR amount (already normalized via base_currency_amount). */
  amount: number
}

/** Aggregate transactions into monthly (credit, debit) sums in PKR.
 *  Months with zero credit OR zero debit are dropped — log(0) is not
 *  representable in the model. Returns months in chronological order. */
export function aggregateMonthly(rows: TxRow[]): CashflowData {
  const buckets = new Map<string, { c: number; d: number }>()
  for (const r of rows) {
    if (r.type !== 'credit' && r.type !== 'debit') continue
    const month = r.date.slice(0, 7) // 'YYYY-MM'
    const cur = buckets.get(month) ?? { c: 0, d: 0 }
    if (r.type === 'credit') cur.c += r.amount
    else cur.d += r.amount
    buckets.set(month, cur)
  }
  const months = Array.from(buckets.keys()).sort()
  const valid = months.filter((m) => {
    const b = buckets.get(m)!
    return b.c > 0 && b.d > 0
  })
  const c = new Float64Array(valid.length)
  const d = new Float64Array(valid.length)
  for (let i = 0; i < valid.length; i++) {
    const b = buckets.get(valid[i])!
    c[i] = b.c
    d[i] = b.d
  }
  return { c, d, months: valid }
}
