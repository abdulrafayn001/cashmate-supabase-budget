// _shared/mcmc/linalg_test.ts
// ============================================================================
// Cholesky + triangular-solve sanity tests.
// ============================================================================

import { assertAlmostEquals, assertEquals } from 'jsr:@std/assert@^1'
import {
  cholesky,
  choleskySolve,
  choleskyWithJitter,
  covarianceFromCorr,
  matvec,
  solveLowerTri,
} from './linalg.ts'

Deno.test('cholesky: 2×2 hand-checked', () => {
  // A = [[4, 2], [2, 5]]; expected L = [[2, 0], [1, 2]]
  const A = new Float64Array([4, 2, 2, 5])
  const L = new Float64Array(4)
  assertEquals(cholesky(A, 2, L), true)
  assertAlmostEquals(L[0], 2, 1e-12)
  assertAlmostEquals(L[1], 0, 1e-12)
  assertAlmostEquals(L[2], 1, 1e-12)
  assertAlmostEquals(L[3], 2, 1e-12)
})

Deno.test('cholesky: rejects non-PSD', () => {
  // A = [[1, 2], [2, 1]] — eigenvalues 3 and -1 → not PSD.
  const A = new Float64Array([1, 2, 2, 1])
  const L = new Float64Array(4)
  assertEquals(cholesky(A, 2, L), false)
})

Deno.test('cholesky: 3×3 reconstructs A = L · Lᵀ', () => {
  const A = new Float64Array([
    25, 15, -5,
    15, 18, 0,
    -5, 0, 11,
  ])
  const L = new Float64Array(9)
  assertEquals(cholesky(A, 3, L), true)
  // Verify L · Lᵀ ≈ A
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0
      for (let k = 0; k < 3; k++) s += L[i * 3 + k] * L[j * 3 + k]
      assertAlmostEquals(s, A[i * 3 + j], 1e-12)
    }
  }
})

Deno.test('solveLowerTri: L · x = b', () => {
  // L = [[2, 0], [1, 2]], b = [4, 5] → x = [2, 1.5]
  const L = new Float64Array([2, 0, 1, 2])
  const b = new Float64Array([4, 5])
  const x = solveLowerTri(L, b, 2)
  assertAlmostEquals(x[0], 2, 1e-12)
  assertAlmostEquals(x[1], 1.5, 1e-12)
})

Deno.test('choleskySolve: A · x = b round-trip', () => {
  const A = new Float64Array([4, 2, 2, 5])
  const L = new Float64Array(4)
  cholesky(A, 2, L)
  const b = new Float64Array([6, 7])
  const x = choleskySolve(L, b, 2)
  // Recompute A·x and compare to b
  const bHat = matvec(A, x, 2)
  assertAlmostEquals(bHat[0], b[0], 1e-12)
  assertAlmostEquals(bHat[1], b[1], 1e-12)
})

Deno.test('covarianceFromCorr: builds Σ = diag(σ) R diag(σ)', () => {
  // σ = [2, 3], R = [[1, 0.5], [0.5, 1]] → Σ = [[4, 3], [3, 9]]
  const sigma = new Float64Array([2, 3])
  const R = new Float64Array([1, 0.5, 0.5, 1])
  const Sigma = covarianceFromCorr(sigma, R, 2)
  assertAlmostEquals(Sigma[0], 4, 1e-12)
  assertAlmostEquals(Sigma[1], 3, 1e-12)
  assertAlmostEquals(Sigma[2], 3, 1e-12)
  assertAlmostEquals(Sigma[3], 9, 1e-12)
})

Deno.test('choleskyWithJitter: recovers from numerical-rank-deficient input', () => {
  // A near-singular: rank-1 outer product + zero — cholesky will fail.
  const A = new Float64Array([1, 1, 1, 1])
  const result = choleskyWithJitter(A, 2)
  if (!result) throw new Error('jittered cholesky did not converge')
  // Jitter > 0 was needed; verify L is now upper-bound-respecting
  if (!(result.jitter > 0)) throw new Error('expected nonzero jitter')
})
