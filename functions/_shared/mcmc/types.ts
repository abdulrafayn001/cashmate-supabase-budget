// _shared/mcmc/types.ts
// ============================================================================
// Shared types for the CashMate MCMC suite (Phase 1 foundations).
// Kept dependency-free so every other file in this folder can import from here
// without pulling in Supabase/Deno globals.
// ============================================================================

export type Rng = () => number

/** A target distribution for a Metropolis sampler. logp must be finite for
 *  the initial point; -Infinity is allowed elsewhere (treated as zero
 *  probability). theta is a Float64Array of length `dim`. */
export interface Target {
  logp(theta: Float64Array): number
  dim: number
}

/** Options for the adaptive Metropolis-Hastings sampler. */
export interface AdaptiveMHOpts {
  nIter: number
  nBurn: number
  thin: number
  initial: Float64Array
  /** Target acceptance rate. Defaults to 0.234 (d>5) or 0.44 (d<=5). */
  targetAccept?: number
  rng: Rng
}

export interface SamplerResult {
  /** Flattened post-burn-in, post-thin draws. Length = nKeep * dim. */
  draws: Float64Array
  /** Acceptance rate over the entire run (incl. burn-in). */
  acceptRate: number
  /** Per-dim final proposal scale (after burn-in adaptation). */
  scale: Float64Array
}

/** Per-parameter posterior summary. Mirrors the §3 ParamSummary contract. */
export interface ParamSummary {
  mean: number
  sd: number
  q05: number
  q25: number
  q50: number
  q75: number
  q95: number
  ess: number
  rhat: number
}

/** Diagnostics block surfaced via the run record + sync response. */
export interface RunDiagnostics {
  rhatMax: number
  essMin: number
  nDivergences: number
  pass: boolean
}

/** What goes into mcmc_runs at finalization. Subset of the SQL columns; the
 *  registry helper fills the rest.
 *
 *  `summary` is intentionally typed as `Record<string, unknown>` here because
 *  every feature shapes it differently (A1 ships a forecast block, A2 ships
 *  per-category posterior densities, B1 ships weight CIs). The `ParamSummary`
 *  type stays as the canonical shape for *global* per-parameter rows; features
 *  use it inside their own typed sub-objects. */
export interface FinalizedRun {
  nSamples: number
  runtimeMs: number
  diagnostics: RunDiagnostics
  summary: Record<string, unknown>
}

export type Feature =
  | 'cashflow_forecast'
  | 'budget_posterior'
  | 'goal_probability'
  | 'portfolio_alloc'
  | 'zakat_forecast'
  | 'subscription_changepoint'
