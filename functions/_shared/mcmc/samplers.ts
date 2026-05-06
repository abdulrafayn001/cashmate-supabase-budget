// _shared/mcmc/samplers.ts
// ============================================================================
// Sampler primitives:
//   - adaptiveMH: random-walk Metropolis with Haario-2001-lite diagonal scaling
//     during burn-in. Adaptation stops at iter == nBurn to preserve ergodicity.
//   - gibbs: composes per-coordinate stepFns into one full-conditional sweep.
//   - slice: Neal-2003 1-D slice sampler with stepping-out + shrinkage.
//
// Inner loops are allocation-frugal — proposal buffers are reused across iters.
// ============================================================================

import type { AdaptiveMHOpts, Rng, SamplerResult, Target } from './types.ts'
import { stdNormal } from './random.ts'

// ---------------------------------------------------------------------------
// Adaptive Metropolis-Hastings (random-walk, diagonal proposal)
// ---------------------------------------------------------------------------
export function adaptiveMH(target: Target, o: AdaptiveMHOpts): SamplerResult {
  const d = target.dim
  const targetAcc = o.targetAccept ?? (d > 5 ? 0.234 : 0.44)

  const theta = new Float64Array(o.initial)
  const prop = new Float64Array(d)
  let logp = target.logp(theta)
  if (!isFinite(logp)) {
    throw new Error('adaptiveMH: initial logp is not finite — bad init or target')
  }

  // Per-coord proposal scale. Adapt during burn-in only.
  const scale = new Float64Array(d).fill(0.1)

  const nKeep = Math.max(0, Math.floor((o.nIter - o.nBurn) / o.thin))
  const draws = new Float64Array(nKeep * d)
  let accepts = 0
  let kept = 0

  for (let i = 0; i < o.nIter; i++) {
    // Propose: theta' = theta + scale * z (z ~ N(0, I))
    for (let k = 0; k < d; k++) prop[k] = theta[k] + scale[k] * stdNormal(o.rng)
    const lpNew = target.logp(prop)
    const logA = lpNew - logp
    if (isFinite(lpNew) && (logA > 0 || Math.log(o.rng()) < logA)) {
      theta.set(prop)
      logp = lpNew
      accepts++
    }

    // Adapt during burn-in: nudge scale toward target acceptance.
    if (i < o.nBurn && i > 50 && i % 50 === 0) {
      const accRate = accepts / (i + 1)
      const factor = Math.exp(0.05 * (accRate - targetAcc))
      for (let k = 0; k < d; k++) scale[k] *= factor
    }

    // Store post-burn draws on the thinning grid.
    if (i >= o.nBurn && (i - o.nBurn) % o.thin === 0 && kept < nKeep) {
      draws.set(theta, kept * d)
      kept++
    }
  }

  return { draws, acceptRate: accepts / o.nIter, scale }
}

// ---------------------------------------------------------------------------
// Gibbs composer: each step is a function that mutates `state` in place.
// The driver runs them in order, post-burn-in samples of `state` are stored.
// ---------------------------------------------------------------------------
export type GibbsStep = (state: Float64Array, rng: Rng) => void

export interface GibbsOpts {
  nIter: number
  nBurn: number
  thin: number
  initial: Float64Array
  steps: GibbsStep[]
  rng: Rng
}

export function gibbs(o: GibbsOpts): { draws: Float64Array } {
  const d = o.initial.length
  const state = new Float64Array(o.initial)
  const nKeep = Math.max(0, Math.floor((o.nIter - o.nBurn) / o.thin))
  const draws = new Float64Array(nKeep * d)
  let kept = 0
  for (let i = 0; i < o.nIter; i++) {
    for (const step of o.steps) step(state, o.rng)
    if (i >= o.nBurn && (i - o.nBurn) % o.thin === 0 && kept < nKeep) {
      draws.set(state, kept * d)
      kept++
    }
  }
  return { draws }
}

// ---------------------------------------------------------------------------
// 1-D slice sampler (Neal 2003) with stepping-out + shrinkage. Used for
// half-Cauchy scale parameters whose full conditional has no closed form.
// `unnormLogP` need only be proportional to the target.
// ---------------------------------------------------------------------------
export function slice(
  unnormLogP: (x: number) => number,
  x0: number,
  rng: Rng,
  width = 1.0,
  maxSteps = 32,
): number {
  const lp0 = unnormLogP(x0)
  if (!isFinite(lp0)) {
    throw new Error('slice: initial logp not finite')
  }
  const logy = lp0 + Math.log(rng())

  // Stepping out: find an interval [L, R] containing the slice.
  let u = rng()
  let L = x0 - width * u
  let R = L + width
  let j = Math.floor(maxSteps * rng())
  let k = maxSteps - 1 - j
  while (j > 0 && unnormLogP(L) > logy) {
    L -= width
    j--
  }
  while (k > 0 && unnormLogP(R) > logy) {
    R += width
    k--
  }

  // Shrinkage: sample uniform on [L, R], shrink on rejection until accepted.
  while (true) {
    const x1 = L + (R - L) * rng()
    const lp1 = unnormLogP(x1)
    if (isFinite(lp1) && lp1 > logy) return x1
    if (x1 < x0) L = x1
    else R = x1
  }
}
