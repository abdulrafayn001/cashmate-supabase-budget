// _shared/mcmc/diagnostics_test.ts
// ============================================================================
// Validates split-Rhat and ESS on synthetic chains where the answer is known.
//
// Cases:
//  1. IID standard normal across 4 chains × 2000 → Rhat ≈ 1.0, ESS ≈ M*N
//  2. Drifting-mean chains (one chain has mean +1, others 0) → Rhat ≫ 1.0
//  3. AR(1) with ρ=0.9 → ESS far below M*N (~ M*N * (1-ρ)/(1+ρ))
// ============================================================================

import { assertEquals } from 'jsr:@std/assert@^1'
import { ess, splitRhat } from './diagnostics.ts'
import { makeRng, stdNormal } from './random.ts'

function buildIidChains(M: number, N: number, dim: number, seed: string): Float64Array[] {
  const r = makeRng(seed)
  const chains: Float64Array[] = []
  for (let c = 0; c < M; c++) {
    const buf = new Float64Array(N * dim)
    for (let i = 0; i < N; i++) {
      for (let d = 0; d < dim; d++) buf[i * dim + d] = stdNormal(r)
    }
    chains.push(buf)
  }
  return chains
}

Deno.test('split-Rhat: IID chains → Rhat ≈ 1.0', () => {
  const chains = buildIidChains(4, 2000, 1, 'rhat-iid')
  const rhat = splitRhat(chains, 0, 1)
  // Tight bar: 4×2000 IID chains should land Rhat well under 1.02.
  if (!(rhat > 0.98 && rhat < 1.02)) {
    throw new Error(`expected Rhat ≈ 1.0, got ${rhat}`)
  }
})

Deno.test('split-Rhat: drifting chain → Rhat noticeably > 1', () => {
  // One chain offset by +3 (3σ); should easily blow past Rhat = 1.5. With +1
  // (1σ) the analytic answer is ~1.10, which is right but not a sharp signal;
  // 3σ drift is unmistakable and isolates the formula from MC noise.
  const chains = buildIidChains(4, 2000, 1, 'rhat-drift')
  for (let i = 0; i < chains[0].length; i++) chains[0][i] += 3
  const rhat = splitRhat(chains, 0, 1)
  if (!(rhat > 1.5)) {
    throw new Error(`expected Rhat > 1.5 for 3σ drift, got ${rhat}`)
  }
})

Deno.test('ESS: IID 4×2000 chains → ESS close to 8000', () => {
  const chains = buildIidChains(4, 2000, 1, 'ess-iid')
  const e = ess(chains, 0, 1)
  // Geyer's combiner is conservative; for IID we expect ESS to land near total
  // sample count. Allow ±20% slack.
  if (!(e > 6500 && e < 9500)) {
    throw new Error(`expected ESS ≈ 8000, got ${e}`)
  }
})

Deno.test('ESS: AR(1) ρ=0.9 → ESS reduced by (1-ρ)/(1+ρ) factor', () => {
  // Build M=4 AR(1) chains with ρ=0.9, stationary variance = 1.
  const M = 4
  const N = 4000
  const rho = 0.9
  const sigEps = Math.sqrt(1 - rho * rho)
  const r = makeRng('ess-ar1')
  const chains: Float64Array[] = []
  for (let c = 0; c < M; c++) {
    const buf = new Float64Array(N)
    let x = stdNormal(r)
    for (let i = 0; i < N; i++) {
      x = rho * x + sigEps * stdNormal(r)
      buf[i] = x
    }
    chains.push(buf)
  }
  const e = ess(chains, 0, 1)
  // Theoretical reduction: (1-ρ)/(1+ρ) = 0.0526; expected ESS ≈ M*N * 0.0526
  // ≈ 842. Allow 50–200% spread (Geyer can over- or under-estimate by 2x).
  if (!(e > 200 && e < 3000)) {
    throw new Error(`expected ESS in [200, 3000] for AR(1) ρ=0.9, got ${e}`)
  }
  // Importantly, MUCH less than IID 16000.
  if (e > 5000) {
    throw new Error(`AR(1) ESS too close to IID — Geyer trim broken? got ${e}`)
  }
})

Deno.test('split-Rhat: rejects single-chain input gracefully', () => {
  const chains = buildIidChains(1, 1000, 1, 'rhat-single')
  const rhat = splitRhat(chains, 0, 1)
  // We return NaN (rather than crash) when M < 2 — caller decides what to do.
  assertEquals(Number.isNaN(rhat), true)
})
