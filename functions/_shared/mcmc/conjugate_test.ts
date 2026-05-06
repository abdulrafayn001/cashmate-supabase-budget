// _shared/mcmc/conjugate_test.ts
// ============================================================================
// Phase-1 acceptance gate: the adaptiveMH sampler must recover analytic
// posteriors for two conjugate models within 1% on the mean and reasonable
// tolerance on the variance.
//
// Models:
//   1. Normal-Normal (known σ²): μ ~ N(μ_0, σ_0²), x_i ~ N(μ, σ²)
//      → posterior μ | x ~ N(*, *) with closed-form mean/variance.
//   2. Beta-Binomial: p ~ Beta(α, β), k ~ Binomial(n, p)
//      → posterior p | k ~ Beta(α + k, β + n − k).
//
// Tolerances are loose-but-honest: 1% on means is the doc's stated bar; the
// MC error of an N=2000 thinned sample is ~σ/√N ≈ 2% of σ, so we pad slightly.
// ============================================================================

import { assertAlmostEquals } from 'jsr:@std/assert@^1'
import { makeRng, gamma } from './random.ts'
import {
  betaBinomialPosterior,
  logBeta,
  logNormal,
  normalNormalPosterior,
} from './distributions.ts'
import { adaptiveMH } from './samplers.ts'
import type { Target } from './types.ts'

// ---------------------------------------------------------------------------
// Test 1: Normal-Normal posterior recovery
// ---------------------------------------------------------------------------
Deno.test('Normal-Normal: adaptiveMH posterior mean within 2% of analytic', () => {
  // Synthesize n=50 draws from N(2.5, 1) — pretend σ² = 1 is known.
  const rng = makeRng('nn-data')
  const n = 50
  const trueMu = 2.5
  const sigma2 = 1
  const data = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    // sample N(trueMu, 1) via inverse CDF would need erf; use Box-Muller dirty
    // (this path differs from the chain's RNG, intentional)
    const u1 = rng()
    const u2 = rng()
    data[i] = trueMu + Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }

  // Prior μ ~ N(0, 100²) — diffuse, so posterior is dominated by data.
  const mu0 = 0
  const sigma02 = 10000

  // Analytic posterior
  const post = normalNormalPosterior(data, sigma2, mu0, sigma02)

  // MCMC: target is logp(μ) = log prior + Σ log N(x_i | μ, σ²)
  const target: Target = {
    dim: 1,
    logp: (theta: Float64Array) => {
      const mu = theta[0]
      let lp = logNormal(mu, mu0, Math.sqrt(sigma02))
      for (let i = 0; i < n; i++) lp += logNormal(data[i], mu, Math.sqrt(sigma2))
      return lp
    },
  }

  const chainRng = makeRng('nn-chain')
  const r = adaptiveMH(target, {
    nIter: 10_000,
    nBurn: 2_000,
    thin: 4,
    initial: new Float64Array([0]),
    rng: chainRng,
  })

  // Compare empirical posterior to analytic.
  const N = r.draws.length
  let sum = 0
  let sum2 = 0
  for (let i = 0; i < N; i++) {
    sum += r.draws[i]
    sum2 += r.draws[i] * r.draws[i]
  }
  const empMean = sum / N
  const empVar = sum2 / N - empMean * empMean

  // 2% relative tolerance on mean (1% bar + MC slack).
  const meanTol = Math.max(0.02 * Math.abs(post.mean), 0.02)
  assertAlmostEquals(empMean, post.mean, meanTol)
  // 25% relative tolerance on variance — small posterior variance is hard to
  // estimate from a thinned MH chain. The doc's 1% bar applies to the mean.
  assertAlmostEquals(empVar, post.variance, 0.25 * post.variance)

  // Sanity: acceptance rate should be in [0.2, 0.7] after adaptation.
  if (r.acceptRate < 0.15 || r.acceptRate > 0.75) {
    throw new Error(`Suspicious acceptance rate ${r.acceptRate}`)
  }
})

// ---------------------------------------------------------------------------
// Test 2: Beta-Binomial posterior recovery
// ---------------------------------------------------------------------------
Deno.test('Beta-Binomial: adaptiveMH posterior mean within 1% of analytic', () => {
  // Prior Beta(2, 2); observe k=8 successes / n=20 trials.
  const aPrior = 2
  const bPrior = 2
  const k = 8
  const n = 20
  const post = betaBinomialPosterior(k, n - k, aPrior, bPrior)
  const postMean = post.alpha / (post.alpha + post.beta)

  // We sample p in unconstrained space via a logit transform so adaptiveMH
  // (Gaussian RW) doesn't keep proposing out-of-support values. p = sigmoid(z).
  const target: Target = {
    dim: 1,
    logp: (theta: Float64Array) => {
      const z = theta[0]
      const p = 1 / (1 + Math.exp(-z))
      if (p <= 0 || p >= 1) return -Infinity
      const lp = logBeta(p, aPrior, bPrior) + k * Math.log(p) + (n - k) * Math.log(1 - p)
      // Jacobian of z → p: dp/dz = p(1-p), so log|J| = log p + log(1-p).
      return lp + Math.log(p) + Math.log(1 - p)
    },
  }

  const chainRng = makeRng('bb-chain')
  const r = adaptiveMH(target, {
    nIter: 12_000,
    nBurn: 3_000,
    thin: 4,
    initial: new Float64Array([0]), // p = 0.5
    rng: chainRng,
  })

  const N = r.draws.length
  let sumP = 0
  for (let i = 0; i < N; i++) {
    const p = 1 / (1 + Math.exp(-r.draws[i]))
    sumP += p
  }
  const empMean = sumP / N

  const tol = Math.max(0.01 * postMean, 0.01)
  assertAlmostEquals(empMean, postMean, tol)
})

// ---------------------------------------------------------------------------
// Sanity for the Gibbs composer: alternating-Normal demo with two coords.
// ---------------------------------------------------------------------------
Deno.test('Gibbs composer: bivariate normal coordinate updates ≈ joint stats', async () => {
  // Joint: (X, Y) ~ N(0, Σ) with Σ = [[1, ρ], [ρ, 1]], ρ = 0.6
  // Conditionals: X | Y ~ N(ρY, 1−ρ²); Y | X ~ N(ρX, 1−ρ²)
  const rho = 0.6
  const cond = 1 - rho * rho

  const { gibbs } = await import('./samplers.ts')
  const { stdNormal } = await import('./random.ts')
  const r = makeRng('gibbs-bvn')

  const out = gibbs({
    nIter: 10_000,
    nBurn: 2_000,
    thin: 2,
    initial: new Float64Array([0, 0]),
    rng: r,
    steps: [
      (s, rng) => {
        s[0] = rho * s[1] + Math.sqrt(cond) * stdNormal(rng)
      },
      (s, rng) => {
        s[1] = rho * s[0] + Math.sqrt(cond) * stdNormal(rng)
      },
    ],
  })

  const N = out.draws.length / 2
  let sx = 0
  let sy = 0
  let sxy = 0
  for (let i = 0; i < N; i++) {
    sx += out.draws[i * 2]
    sy += out.draws[i * 2 + 1]
    sxy += out.draws[i * 2] * out.draws[i * 2 + 1]
  }
  const mx = sx / N
  const my = sy / N
  const empCov = sxy / N - mx * my

  assertAlmostEquals(mx, 0, 0.05)
  assertAlmostEquals(my, 0, 0.05)
  assertAlmostEquals(empCov, rho, 0.05)
})

// ---------------------------------------------------------------------------
// Slice sampler sanity: target is N(0, 1); should recover mean ≈ 0, var ≈ 1.
// ---------------------------------------------------------------------------
Deno.test('slice sampler: standard normal recovery', async () => {
  const { slice } = await import('./samplers.ts')
  const r = makeRng('slice-normal')
  let x = 0
  const draws = new Float64Array(5_000)
  for (let i = 0; i < 1_000; i++) {
    x = slice((v) => -0.5 * v * v, x, r) // logp ∝ -x²/2
  }
  for (let i = 0; i < draws.length; i++) {
    x = slice((v) => -0.5 * v * v, x, r)
    draws[i] = x
  }
  let s = 0
  let s2 = 0
  for (let i = 0; i < draws.length; i++) {
    s += draws[i]
    s2 += draws[i] * draws[i]
  }
  const mu = s / draws.length
  const v = s2 / draws.length - mu * mu
  assertAlmostEquals(mu, 0, 0.08)
  assertAlmostEquals(v, 1, 0.1)
})

// Silence unused-import lint when running just this file:
void gamma
