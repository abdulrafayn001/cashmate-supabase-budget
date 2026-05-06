// _shared/mcmc/features/budget.ts
// ============================================================================
// A2 — Bayesian budget posterior.
//
// Model (per design doc §2 A2 + audit-patch G2):
//
//   Two independent hierarchies, expense-side only (income tracked elsewhere).
//   For category c with parent_group g(c) ∈ {needs, wants, savings}:
//
//     log(x_{c,m}) ~ N(θ_c, σ_c²)               # x_{c,m} = monthly debit total
//     θ_c | μ_g, τ_g ~ N(μ_g, τ_g²)             # partial pooling within group
//     σ_c ~ HC+(0.5)
//     μ_g ~ N(log(median_g), 2)
//     τ_g ~ HC+(0.5)
//
// Sampler: pure Gibbs via the Wand 2011 half-Cauchy → IG augmentation.
// Introducing aux a_c (per category) and b_g (per group):
//
//     σ_c² | rest ~ IG((n_c+1)/2, S_c/2 + 1/a_c)        S_c = Σ_m (log x_{c,m} − θ_c)²
//     a_c    | σ_c² ~ IG(1, 1/s_σ² + 1/σ_c²)
//     τ_g²   | rest ~ IG((K_g+1)/2, T_g/2 + 1/b_g)      T_g = Σ_{c ∈ g}(θ_c − μ_g)²
//     b_g    | τ_g² ~ IG(1, 1/s_τ² + 1/τ_g²)
//     θ_c    | rest ~ N(post_mean, post_var)             [Normal-Normal]
//     μ_g    | rest ~ N(post_mean, post_var)             [Normal-Normal]
//
// Every conditional is closed-form; no MH steps, no tuning. ~30 categories ×
// 12 months data → runtime budget < 2 s.
// ============================================================================

import type { Rng } from '../types.ts'
import { gamma, inverseGamma, makeRng, stdNormal } from '../random.ts'
import { ess, splitRhat, summarize } from '../diagnostics.ts'

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type ParentGroup = 'needs' | 'wants' | 'savings'

const GROUPS: ParentGroup[] = ['needs', 'wants', 'savings']

/** A budgeted category as the model sees it. */
export interface CategoryMeta {
  id: string
  name: string
  parentGroup: ParentGroup
  /** category_budgets.allocated_amount, PKR base currency. */
  allocated: number
}

/** Raw transaction row pulled by the Edge Function (post-RLS). */
export interface TxRow {
  /** ISO datetime. */
  date: string
  /** category_id (must match a CategoryMeta.id; rows with other categories are dropped). */
  categoryId: string
  /** PKR amount (base_currency_amount, falling back to source_amount upstream). */
  amount: number
}

/** Period metadata for the active master_budget row. */
export interface BudgetPeriodMeta {
  startDate: string  // ISO
  endDate: string    // ISO
  /** Today's ISO date — passed in by caller to keep the function deterministic. */
  today: string
  /** Per-category actuals already realised in [startDate, today]. */
  spentSoFar: Record<string, number>
}

/** Aggregated input the sampler consumes. */
export interface BudgetData {
  categories: CategoryMeta[]
  /** Per-category log monthly totals. monthly[c][i] = log(Σ debits in cat c, month i). */
  monthly: Float64Array[]
  /** Pre-computed sufficient stat: per-category Σ log(x_{c,m}). */
  sumLogX: Float64Array
  /** Per-category Σ (log x_{c,m})². Lets us compute S_c in O(1) per iter. */
  sumLogX2: Float64Array
  /** Per-category n_c. */
  nObs: Int32Array
  /** Per-category log(median monthly total). Used for theta init + group prior fallback. */
  logMedian: Float64Array
  /** Per-group log-median across the group (initial μ_g). */
  groupLogMedian: Record<ParentGroup, number>
  /** ISO month labels covered by the data, sorted. */
  months: string[]
}

export interface BudgetOpts {
  nIter: number
  nBurn: number
  thin: number
  nChains: number
  yieldEvery: number
}

export const DEFAULT_OPTS: BudgetOpts = {
  nIter: 3000,
  nBurn: 1500,
  thin: 2,
  nChains: 4,
  yieldEvery: 250,
}

/** Half-Cauchy prior scales (per design doc §2 A2). */
const S_SIGMA = 0.5
const S_TAU = 0.5
/** μ_g prior std = 2  →  variance = 4. */
const MU_PRIOR_VAR = 4

export interface ForecastQuantiles {
  median: number
  q05: number
  q25: number
  q75: number
  q95: number
}

export interface CategoryPosterior {
  categoryId: string
  categoryName: string
  parentGroup: ParentGroup
  /** Posterior on E[x_{c,·}] = exp(θ_c + 0.5 σ_c²) — the "true monthly mean spend" (PKR). */
  monthlyMean: ForecastQuantiles
  /** Posterior on the period total = spentSoFar + simulated remainder (PKR). */
  periodTotal: ForecastQuantiles
  /** P(periodTotal > allocated). */
  pOver: number
  allocated: number
  spentSoFar: number
  nObservations: number
}

export interface GroupPosterior {
  group: ParentGroup
  /** Posterior on group period total (sum across categories in the group). */
  periodTotal: ForecastQuantiles
  pOver: number
  /** Allocated total = Σ category allocations in the group. */
  allocated: number
  spentSoFar: number
}

export interface BudgetResult {
  diagnostics: {
    rhatMax: number
    essMin: number
    nDivergences: number
    pass: boolean
    perParam: Record<string, { rhat: number; ess: number }>
  }
  summary: Record<string, ReturnType<typeof summarize>>
  posteriors: CategoryPosterior[]
  groupTotals: GroupPosterior[]
  period: { startDate: string; endDate: string; daysRemaining: number; periodDays: number }
  runtimeMs: number
  nSamples: number
}

// ---------------------------------------------------------------------------
// State layout: one Float64Array per chain, packed as
//
//   [θ_0..θ_{C-1}] [σ²_0..σ²_{C-1}] [a_0..a_{C-1}]
//   [μ_needs μ_wants μ_savings]
//   [τ²_needs τ²_wants τ²_savings]
//   [b_needs b_wants b_savings]
//
// Total dim = 3*C + 9.
// ---------------------------------------------------------------------------

interface Layout {
  C: number
  G: number
  thetaStart: number
  sigma2Start: number
  aStart: number
  muStart: number
  tau2Start: number
  bStart: number
  dim: number
}

function makeLayout(C: number): Layout {
  const G = GROUPS.length
  return {
    C,
    G,
    thetaStart: 0,
    sigma2Start: C,
    aStart: 2 * C,
    muStart: 3 * C,
    tau2Start: 3 * C + G,
    bStart: 3 * C + 2 * G,
    dim: 3 * C + 3 * G,
  }
}

function paramNames(data: BudgetData): string[] {
  const names: string[] = []
  for (const c of data.categories) names.push(`theta[${c.id}]`)
  for (const c of data.categories) names.push(`sigma2[${c.id}]`)
  for (const c of data.categories) names.push(`a[${c.id}]`)
  for (const g of GROUPS) names.push(`mu[${g}]`)
  for (const g of GROUPS) names.push(`tau2[${g}]`)
  for (const g of GROUPS) names.push(`b[${g}]`)
  return names
}

function groupIndex(g: ParentGroup): number {
  return GROUPS.indexOf(g)
}

// ---------------------------------------------------------------------------
// Initialization — median-anchored, mildly informative.
// ---------------------------------------------------------------------------
function initState(data: BudgetData, layout: Layout, rng: Rng): Float64Array {
  const state = new Float64Array(layout.dim)
  for (let c = 0; c < layout.C; c++) {
    state[layout.thetaStart + c] = data.logMedian[c] + 0.05 * stdNormal(rng)
    state[layout.sigma2Start + c] = 0.25  // σ ≈ 0.5
    state[layout.aStart + c] = 1.0
  }
  for (let g = 0; g < layout.G; g++) {
    state[layout.muStart + g] = data.groupLogMedian[GROUPS[g]]
    state[layout.tau2Start + g] = 0.25
    state[layout.bStart + g] = 1.0
  }
  return state
}

// ---------------------------------------------------------------------------
// One full Gibbs sweep over the state vector. Mutates `state` in place.
// Order: σ_c², a_c, τ_g², b_g, θ_c, μ_g.
// (σ-block before θ-block so the very first sweep doesn't depend on the seed
// θ values; either order is valid for stationarity.)
// ---------------------------------------------------------------------------
function sweep(state: Float64Array, data: BudgetData, layout: Layout, rng: Rng): void {
  const C = layout.C
  const G = layout.G

  // --- 1. σ_c² | rest ~ IG((n_c+1)/2, S_c/2 + 1/a_c)
  //     where S_c = Σ_m (log x_{c,m} − θ_c)² = sumLogX2_c − 2 θ_c sumLogX_c + n_c θ_c².
  for (let c = 0; c < C; c++) {
    const theta = state[layout.thetaStart + c]
    const a = state[layout.aStart + c]
    const n = data.nObs[c]
    const S = data.sumLogX2[c] - 2 * theta * data.sumLogX[c] + n * theta * theta
    // S can drift ε-negative due to floating-point cancellation; clamp.
    const Sclamped = S > 0 ? S : 0
    const alphaPost = (n + 1) / 2
    const betaPost = Sclamped / 2 + 1 / a
    state[layout.sigma2Start + c] = inverseGamma(rng, alphaPost, betaPost)
  }

  // --- 2. a_c | σ_c² ~ IG(1, 1/s_σ² + 1/σ_c²)
  for (let c = 0; c < C; c++) {
    const sigma2 = state[layout.sigma2Start + c]
    state[layout.aStart + c] = inverseGamma(rng, 1, 1 / (S_SIGMA * S_SIGMA) + 1 / sigma2)
  }

  // --- 3. τ_g² | rest ~ IG((K_g+1)/2, T_g/2 + 1/b_g)
  //     T_g = Σ_{c ∈ g}(θ_c − μ_g)².  K_g = # categories in group g.
  for (let g = 0; g < G; g++) {
    const mu = state[layout.muStart + g]
    const b = state[layout.bStart + g]
    let K = 0
    let T = 0
    for (let c = 0; c < C; c++) {
      if (groupIndex(data.categories[c].parentGroup) !== g) continue
      const d = state[layout.thetaStart + c] - mu
      T += d * d
      K++
    }
    if (K === 0) {
      // No categories in this group → keep τ² near its prior median.
      state[layout.tau2Start + g] = inverseGamma(rng, 0.5, 1 / b)
      continue
    }
    const alphaPost = (K + 1) / 2
    const betaPost = T / 2 + 1 / b
    state[layout.tau2Start + g] = inverseGamma(rng, alphaPost, betaPost)
  }

  // --- 4. b_g | τ_g² ~ IG(1, 1/s_τ² + 1/τ_g²)
  for (let g = 0; g < G; g++) {
    const tau2 = state[layout.tau2Start + g]
    state[layout.bStart + g] = inverseGamma(rng, 1, 1 / (S_TAU * S_TAU) + 1 / tau2)
  }

  // --- 5. θ_c | rest ~ N(post_mean, post_var)
  //     prec_post = 1/τ_g² + n_c/σ_c²
  //     mean_post = post_var · (μ_g/τ_g² + sumLogX_c/σ_c²)
  for (let c = 0; c < C; c++) {
    const g = groupIndex(data.categories[c].parentGroup)
    const tau2 = state[layout.tau2Start + g]
    const mu = state[layout.muStart + g]
    const sigma2 = state[layout.sigma2Start + c]
    const n = data.nObs[c]
    const precPrior = 1 / tau2
    const precData = n / sigma2
    const precPost = precPrior + precData
    const varPost = 1 / precPost
    const meanPost = varPost * (mu * precPrior + data.sumLogX[c] / sigma2)
    state[layout.thetaStart + c] = meanPost + Math.sqrt(varPost) * stdNormal(rng)
  }

  // --- 6. μ_g | rest ~ N(post_mean, post_var)
  //     prec_post = 1/MU_PRIOR_VAR + K_g/τ_g²
  //     mean_post = post_var · (m_g/MU_PRIOR_VAR + (Σ θ_c)/τ_g²)
  for (let g = 0; g < G; g++) {
    const groupName = GROUPS[g]
    const tau2 = state[layout.tau2Start + g]
    let K = 0
    let sumTheta = 0
    for (let c = 0; c < C; c++) {
      if (groupIndex(data.categories[c].parentGroup) !== g) continue
      sumTheta += state[layout.thetaStart + c]
      K++
    }
    const mPrior = data.groupLogMedian[groupName]
    const precPrior = 1 / MU_PRIOR_VAR
    const precData = K / tau2
    const precPost = precPrior + precData
    const varPost = 1 / precPost
    const meanPost = varPost * (mPrior * precPrior + sumTheta / tau2)
    state[layout.muStart + g] = meanPost + Math.sqrt(varPost) * stdNormal(rng)
  }
}

// ---------------------------------------------------------------------------
// Single-chain runner.
// ---------------------------------------------------------------------------
async function runChain(
  data: BudgetData,
  opts: BudgetOpts,
  rng: Rng,
  yieldFn?: () => Promise<void>,
): Promise<Float64Array> {
  const layout = makeLayout(data.categories.length)
  const state = initState(data, layout, rng)

  const nKeep = Math.max(0, Math.floor((opts.nIter - opts.nBurn) / opts.thin))
  const draws = new Float64Array(nKeep * layout.dim)
  let kept = 0

  for (let i = 0; i < opts.nIter; i++) {
    sweep(state, data, layout, rng)
    if (i >= opts.nBurn && (i - opts.nBurn) % opts.thin === 0 && kept < nKeep) {
      draws.set(state, kept * layout.dim)
      kept++
    }
    if (yieldFn && i > 0 && i % opts.yieldEvery === 0) await yieldFn()
  }

  return draws
}

// ---------------------------------------------------------------------------
// Forecast: per category and per group, simulate the period total and compute
// quantiles + P(over).
// ---------------------------------------------------------------------------
function forecastFromDraws(
  allDraws: Float64Array,
  layout: Layout,
  data: BudgetData,
  period: { startDate: string; endDate: string; today: string; spentSoFar: Record<string, number> },
  rng: Rng,
): {
  posteriors: CategoryPosterior[]
  groupTotals: GroupPosterior[]
  periodInfo: { startDate: string; endDate: string; daysRemaining: number; periodDays: number }
} {
  const C = layout.C
  const dim = layout.dim
  const N = allDraws.length / dim

  const startMs = new Date(period.startDate).getTime()
  const endMs = new Date(period.endDate).getTime()
  const todayMs = new Date(period.today).getTime()
  const dayMs = 24 * 60 * 60 * 1000
  const periodDays = Math.max(1, Math.round((endMs - startMs) / dayMs))
  const daysElapsed = Math.max(0, Math.min(periodDays, Math.round((todayMs - startMs) / dayMs)))
  const daysRemaining = Math.max(0, periodDays - daysElapsed)
  // Convert to a fraction of an "average month" (30.4375 days) for the
  // log-normal predictive: the model fits on monthly totals, so to forecast a
  // partial-period remainder we scale a single monthly draw by remainFraction.
  const remainFraction = daysRemaining / 30.4375

  // Per category: collect period totals, monthly means, and pOver.
  const posteriors: CategoryPosterior[] = []
  // Per group: accumulate group totals across draws.
  const groupTotalDraws: Record<ParentGroup, Float64Array> = {
    needs: new Float64Array(N),
    wants: new Float64Array(N),
    savings: new Float64Array(N),
  }
  const groupAllocated: Record<ParentGroup, number> = { needs: 0, wants: 0, savings: 0 }
  const groupSpentSoFar: Record<ParentGroup, number> = { needs: 0, wants: 0, savings: 0 }

  for (let c = 0; c < C; c++) {
    const meta = data.categories[c]
    const allocated = meta.allocated
    const spentSoFar = period.spentSoFar[meta.id] ?? 0
    groupAllocated[meta.parentGroup] += allocated
    groupSpentSoFar[meta.parentGroup] += spentSoFar

    // Per draw: simulate posterior predictive for one month, scale to remainder.
    const monthlyMeanDraws = new Float64Array(N)
    const periodTotalDraws = new Float64Array(N)
    for (let i = 0; i < N; i++) {
      const off = i * dim
      const theta = allDraws[off + layout.thetaStart + c]
      const sigma2 = allDraws[off + layout.sigma2Start + c]
      const sigma = Math.sqrt(Math.max(sigma2, 1e-12))
      // Posterior on monthly mean (E[x|θ,σ²] = exp(θ + σ²/2))
      monthlyMeanDraws[i] = Math.exp(theta + 0.5 * sigma2)
      // Posterior predictive draw of one month's total
      const monthDraw = Math.exp(theta + sigma * stdNormal(rng))
      const remainder = monthDraw * remainFraction
      const periodTotal = spentSoFar + remainder
      periodTotalDraws[i] = periodTotal
      groupTotalDraws[meta.parentGroup][i] += periodTotal
    }

    let nOver = 0
    for (let i = 0; i < N; i++) if (periodTotalDraws[i] > allocated) nOver++

    posteriors.push({
      categoryId: meta.id,
      categoryName: meta.name,
      parentGroup: meta.parentGroup,
      monthlyMean: quantiles(monthlyMeanDraws),
      periodTotal: quantiles(periodTotalDraws),
      pOver: N > 0 ? nOver / N : 0,
      allocated,
      spentSoFar,
      nObservations: data.nObs[c],
    })
  }

  // Group rollups.
  const groupTotals: GroupPosterior[] = []
  for (const g of GROUPS) {
    if (groupAllocated[g] === 0) continue  // skip groups with no budgeted categories
    const draws = groupTotalDraws[g]
    let nOver = 0
    for (let i = 0; i < N; i++) if (draws[i] > groupAllocated[g]) nOver++
    groupTotals.push({
      group: g,
      periodTotal: quantiles(draws),
      pOver: N > 0 ? nOver / N : 0,
      allocated: groupAllocated[g],
      spentSoFar: groupSpentSoFar[g],
    })
  }

  return {
    posteriors,
    groupTotals,
    periodInfo: { startDate: period.startDate, endDate: period.endDate, daysRemaining, periodDays },
  }
}

function quantiles(x: Float64Array): ForecastQuantiles {
  if (x.length === 0) return { median: 0, q05: 0, q25: 0, q75: 0, q95: 0 }
  const sorted = Array.from(x).sort((a, b) => a - b)
  const q = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))]
  return { median: q(0.5), q05: q(0.05), q25: q(0.25), q75: q(0.75), q95: q(0.95) }
}

// ---------------------------------------------------------------------------
// Multi-chain driver — runs nChains independent chains, computes diagnostics,
// summarizes globals, builds the per-category and per-group forecasts.
// ---------------------------------------------------------------------------
export interface RunBudgetOpts extends Partial<BudgetOpts> {
  /** Seed prefix; chain index appended ('seed:0', ...). */
  seed: string
  yieldFn?: () => Promise<void>
}

export async function runBudget(
  data: BudgetData,
  period: BudgetPeriodMeta,
  opts: RunBudgetOpts,
): Promise<BudgetResult> {
  const merged: BudgetOpts = { ...DEFAULT_OPTS, ...opts }
  const layout = makeLayout(data.categories.length)
  const yieldFn = opts.yieldFn ?? (() => new Promise<void>((r) => setTimeout(r, 0)))

  const t0 = Date.now()
  const chains: Float64Array[] = []
  for (let c = 0; c < merged.nChains; c++) {
    const chainRng = makeRng(`${opts.seed}:${c}`)
    const draws = await runChain(data, merged, chainRng, yieldFn)
    chains.push(draws)
  }
  const runtimeMs = Date.now() - t0

  // Diagnostics — gate uses globals only (μ_g, τ²_g). Per-category θ/σ² have
  // wide expected variability and aren't part of the convergence gate.
  const names = paramNames(data)
  const globalStart = layout.muStart
  let rhatMax = 0
  let essMin = Infinity
  const perParam: Record<string, { rhat: number; ess: number }> = {}
  for (let p = 0; p < layout.dim; p++) {
    const r = splitRhat(chains, p, layout.dim)
    const e = ess(chains, p, layout.dim)
    perParam[names[p]] = { rhat: isFinite(r) ? r : NaN, ess: e }
    if (p >= globalStart && p < layout.bStart) {
      // Apply gate to μ_g and τ²_g; aux b_g is uninteresting.
      if (isFinite(r) && r > rhatMax) rhatMax = r
      if (isFinite(e) && e < essMin) essMin = e
    }
  }

  // Summary for global params (μ_g, τ²_g).
  const summary: BudgetResult['summary'] = {}
  for (let p = layout.muStart; p < layout.bStart; p++) {
    summary[names[p]] = summarize(chains, p, layout.dim)
  }

  // Forecast — pool draws across chains.
  const total = chains.reduce((s, c) => s + c.length, 0)
  const allDraws = new Float64Array(total)
  let off = 0
  for (const c of chains) {
    allDraws.set(c, off)
    off += c.length
  }
  const forecastRng = makeRng(`${opts.seed}:forecast`)
  const { posteriors, groupTotals, periodInfo } = forecastFromDraws(
    allDraws,
    layout,
    data,
    { startDate: period.startDate, endDate: period.endDate, today: period.today, spentSoFar: period.spentSoFar },
    forecastRng,
  )

  const nSamples = (allDraws.length / layout.dim) | 0
  const pass = isFinite(rhatMax) && isFinite(essMin) && rhatMax < 1.01 && essMin > 400
  return {
    diagnostics: { rhatMax, essMin, nDivergences: 0, pass, perParam },
    summary,
    posteriors,
    groupTotals,
    period: periodInfo,
    runtimeMs,
    nSamples,
  }
}

// ---------------------------------------------------------------------------
// Aggregation: rows + categories → BudgetData.
//
// Bucket debits by (category_id, year-month). Categories with zero
// observations are kept in the model so partial pooling can still produce
// a (prior-driven) posterior for them.
// ---------------------------------------------------------------------------
export function aggregateMonthly(rows: TxRow[], categories: CategoryMeta[]): BudgetData {
  const byCatMonth = new Map<string, number>()  // key = `${categoryId}|${YYYY-MM}`
  const allMonths = new Set<string>()
  const knownIds = new Set(categories.map((c) => c.id))

  for (const r of rows) {
    if (!knownIds.has(r.categoryId)) continue
    if (!isFinite(r.amount) || r.amount <= 0) continue
    const month = r.date.slice(0, 7)
    if (month.length !== 7) continue
    const key = `${r.categoryId}|${month}`
    byCatMonth.set(key, (byCatMonth.get(key) ?? 0) + r.amount)
    allMonths.add(month)
  }

  const months = Array.from(allMonths).sort()
  const C = categories.length

  const monthly: Float64Array[] = new Array(C)
  const sumLogX = new Float64Array(C)
  const sumLogX2 = new Float64Array(C)
  const nObs = new Int32Array(C)
  const logMedian = new Float64Array(C)

  // Per-group medians (across categories in the group, of category log-medians).
  const groupBuckets: Record<ParentGroup, number[]> = { needs: [], wants: [], savings: [] }

  for (let c = 0; c < C; c++) {
    const cat = categories[c]
    const observed: number[] = []
    for (const m of months) {
      const v = byCatMonth.get(`${cat.id}|${m}`)
      if (v !== undefined && v > 0) observed.push(Math.log(v))
    }
    const arr = new Float64Array(observed)
    monthly[c] = arr
    let s = 0
    let s2 = 0
    for (let i = 0; i < arr.length; i++) {
      s += arr[i]
      s2 += arr[i] * arr[i]
    }
    sumLogX[c] = s
    sumLogX2[c] = s2
    nObs[c] = arr.length

    // Log-median: prefer observed; fall back to log(allocated) for categories
    // with no data so init/state lands somewhere reasonable.
    if (arr.length > 0) {
      const sorted = Array.from(arr).sort((a, b) => a - b)
      const med = sorted.length % 2
        ? sorted[(sorted.length - 1) / 2]
        : 0.5 * (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2])
      logMedian[c] = med
    } else if (cat.allocated > 0) {
      logMedian[c] = Math.log(cat.allocated)
    } else {
      logMedian[c] = Math.log(1)
    }
    groupBuckets[cat.parentGroup].push(logMedian[c])
  }

  const groupLogMedian: Record<ParentGroup, number> = {
    needs: medianOf(groupBuckets.needs, Math.log(10_000)),
    wants: medianOf(groupBuckets.wants, Math.log(5_000)),
    savings: medianOf(groupBuckets.savings, Math.log(5_000)),
  }

  return {
    categories,
    monthly,
    sumLogX,
    sumLogX2,
    nObs,
    logMedian,
    groupLogMedian,
    months,
  }
}

function medianOf(xs: number[], fallback: number): number {
  if (xs.length === 0) return fallback
  const s = xs.slice().sort((a, b) => a - b)
  const n = s.length
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2])
}

// Re-export the gamma helper so test files don't need a deeper import path.
export { gamma }
