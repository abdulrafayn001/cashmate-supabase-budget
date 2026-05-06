// _shared/mcmc/distributions.ts
// ============================================================================
// Log-PDFs for the distributions the MCMC suite needs. All return -Infinity
// out-of-support; never exponentiate before summing log-probs. Constants are
// included so MH ratios can be verified against analytic posteriors in tests.
// ============================================================================

const LOG_2PI = Math.log(2 * Math.PI)
const LOG_PI = Math.log(Math.PI)

// ---------------------------------------------------------------------------
// log-Gamma function (Lanczos approximation, ~1e-15 absolute accuracy).
// ---------------------------------------------------------------------------
const G_COEFFS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7,
]

export function lgamma(x: number): number {
  if (x < 0.5) {
    // Reflection: Γ(x)Γ(1-x) = π / sin(πx)
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x)
  }
  x -= 1
  let a = 0.99999999999980993
  for (let i = 0; i < G_COEFFS.length; i++) a += G_COEFFS[i] / (x + i + 1)
  const t = x + G_COEFFS.length - 0.5
  return 0.5 * LOG_2PI + (x + 0.5) * Math.log(t) - t + Math.log(a)
}

/** log B(α, β) = log Γ(α) + log Γ(β) − log Γ(α+β) */
export function lbeta(a: number, b: number): number {
  return lgamma(a) + lgamma(b) - lgamma(a + b)
}

// ---------------------------------------------------------------------------
// Univariate log-PDFs
// ---------------------------------------------------------------------------

/** log N(x | μ, σ²) */
export function logNormal(x: number, mu: number, sigma: number): number {
  if (sigma <= 0) return -Infinity
  const z = (x - mu) / sigma
  return -0.5 * (LOG_2PI + 2 * Math.log(sigma) + z * z)
}

/** log LogNormal(x | μ, σ²): x > 0, log(x) ~ N(μ, σ²) */
export function logLogNormal(x: number, mu: number, sigma: number): number {
  if (x <= 0 || sigma <= 0) return -Infinity
  return logNormal(Math.log(x), mu, sigma) - Math.log(x)
}

/** log Student-t(x | μ, σ², ν) */
export function logT(x: number, mu: number, sigma: number, nu: number): number {
  if (sigma <= 0 || nu <= 0) return -Infinity
  const z = (x - mu) / sigma
  return (
    lgamma((nu + 1) / 2) -
    lgamma(nu / 2) -
    0.5 * Math.log(nu * Math.PI) -
    Math.log(sigma) -
    ((nu + 1) / 2) * Math.log(1 + (z * z) / nu)
  )
}

/** log Gamma(x | α, β) (rate parametrization: mean = α/β) */
export function logGamma(x: number, alpha: number, beta: number): number {
  if (x <= 0 || alpha <= 0 || beta <= 0) return -Infinity
  return alpha * Math.log(beta) - lgamma(alpha) + (alpha - 1) * Math.log(x) - beta * x
}

/** log InverseGamma(x | α, β): if 1/X ~ Gamma(α, β), X ~ IG(α, β) */
export function logInverseGamma(x: number, alpha: number, beta: number): number {
  if (x <= 0 || alpha <= 0 || beta <= 0) return -Infinity
  return alpha * Math.log(beta) - lgamma(alpha) - (alpha + 1) * Math.log(x) - beta / x
}

/** log Beta(x | α, β), x ∈ (0, 1) */
export function logBeta(x: number, alpha: number, beta: number): number {
  if (x <= 0 || x >= 1 || alpha <= 0 || beta <= 0) return -Infinity
  return -lbeta(alpha, beta) + (alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x)
}

/** log Half-Cauchy(x | σ) on x > 0 */
export function logHalfCauchy(x: number, sigma: number): number {
  if (x < 0 || sigma <= 0) return -Infinity
  return Math.log(2) - LOG_PI - Math.log(sigma) - Math.log(1 + (x / sigma) * (x / sigma))
}

/** log Dirichlet(p | α), p on the simplex of length K */
export function logDirichlet(p: Float64Array | number[], alpha: Float64Array | number[]): number {
  const K = p.length
  if (alpha.length !== K) return -Infinity
  let logNorm = 0
  let alphaSum = 0
  let logp = 0
  for (let i = 0; i < K; i++) {
    if (p[i] <= 0 || p[i] >= 1 || alpha[i] <= 0) return -Infinity
    alphaSum += alpha[i]
    logNorm += lgamma(alpha[i])
    logp += (alpha[i] - 1) * Math.log(p[i])
  }
  return lgamma(alphaSum) - logNorm + logp
}

// ---------------------------------------------------------------------------
// Conjugate posterior helpers (used by tests + simple Gibbs steps)
// ---------------------------------------------------------------------------

/** Posterior of N(μ | x_1..x_n) under known σ² and prior μ ~ N(μ_0, σ_0²).
 *  Returns the posterior (mean, variance). */
export function normalNormalPosterior(
  data: Float64Array | number[],
  sigma2: number,
  mu0: number,
  sigma02: number,
): { mean: number; variance: number } {
  const n = data.length
  let sumX = 0
  for (let i = 0; i < n; i++) sumX += data[i]
  const variance = 1 / (1 / sigma02 + n / sigma2)
  const mean = variance * (mu0 / sigma02 + sumX / sigma2)
  return { mean, variance }
}

/** Posterior of Beta(p | k successes, n-k failures) under prior Beta(α, β). */
export function betaBinomialPosterior(
  successes: number,
  failures: number,
  alpha: number,
  beta: number,
): { alpha: number; beta: number } {
  return { alpha: alpha + successes, beta: beta + failures }
}
