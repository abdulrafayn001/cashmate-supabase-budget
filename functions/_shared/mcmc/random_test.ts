// _shared/mcmc/random_test.ts
// ============================================================================
// Sanity tests for the PRNG + samplers. Run with:
//   deno test supabase/functions/_shared/mcmc/random_test.ts
// ============================================================================

import { assertAlmostEquals, assertEquals } from 'jsr:@std/assert@^1'
import { beta, cyrb128, dirichlet, gamma, makeRng, sfc32, stdNormal } from './random.ts'

Deno.test('cyrb128 is deterministic and 4×u32', () => {
  const a = cyrb128('user-123:run-abc')
  const b = cyrb128('user-123:run-abc')
  assertEquals(a, b)
  for (const v of a) {
    assertEquals(Number.isInteger(v), true)
    assertEquals(v >= 0 && v <= 0xffffffff, true)
  }
})

Deno.test('sfc32 reproducibility — same seed yields same stream', () => {
  const r1 = sfc32(1, 2, 3, 4)
  const r2 = sfc32(1, 2, 3, 4)
  for (let i = 0; i < 100; i++) assertEquals(r1(), r2())
})

Deno.test('makeRng yields uniform [0, 1)', () => {
  const r = makeRng('uniformity-test')
  let mn = Infinity
  let mx = -Infinity
  for (let i = 0; i < 10_000; i++) {
    const x = r()
    if (x < mn) mn = x
    if (x > mx) mx = x
    if (x < 0 || x >= 1) throw new Error(`out-of-range: ${x}`)
  }
  // After 10k draws we expect to be within ~1e-4 of the bounds.
  if (mn > 0.01 || mx < 0.99) throw new Error(`poor coverage: [${mn}, ${mx}]`)
})

Deno.test('stdNormal: mean ≈ 0, var ≈ 1 over 50k draws', () => {
  const r = makeRng('normal-test')
  const N = 50_000
  let s = 0
  let s2 = 0
  for (let i = 0; i < N; i++) {
    const z = stdNormal(r)
    s += z
    s2 += z * z
  }
  const mu = s / N
  const v = s2 / N - mu * mu
  assertAlmostEquals(mu, 0, 0.05)
  assertAlmostEquals(v, 1, 0.05)
})

Deno.test('gamma(α=2, β=1): mean ≈ 2, var ≈ 2', () => {
  const r = makeRng('gamma-2')
  const N = 30_000
  let s = 0
  let s2 = 0
  for (let i = 0; i < N; i++) {
    const x = gamma(r, 2, 1)
    s += x
    s2 += x * x
  }
  const mu = s / N
  const v = s2 / N - mu * mu
  assertAlmostEquals(mu, 2, 0.1)
  assertAlmostEquals(v, 2, 0.2)
})

Deno.test('gamma(α=0.5): handles shape < 1 (boost path)', () => {
  const r = makeRng('gamma-half')
  const N = 30_000
  let s = 0
  for (let i = 0; i < N; i++) s += gamma(r, 0.5, 1)
  const mu = s / N
  // Mean = α * scale = 0.5
  assertAlmostEquals(mu, 0.5, 0.05)
})

Deno.test('beta(2, 5): mean ≈ 2/7', () => {
  const r = makeRng('beta-test')
  const N = 30_000
  let s = 0
  for (let i = 0; i < N; i++) s += beta(r, 2, 5)
  const mu = s / N
  assertAlmostEquals(mu, 2 / 7, 0.01)
})

Deno.test('dirichlet sums to 1', () => {
  const r = makeRng('dir-test')
  const alpha = new Float64Array([1, 1, 1, 1])
  for (let trial = 0; trial < 100; trial++) {
    const p = dirichlet(r, alpha)
    let s = 0
    for (let i = 0; i < p.length; i++) s += p[i]
    assertAlmostEquals(s, 1, 1e-12)
    for (let i = 0; i < p.length; i++) {
      if (p[i] < 0 || p[i] > 1) throw new Error(`out of simplex: ${p[i]}`)
    }
  }
})
