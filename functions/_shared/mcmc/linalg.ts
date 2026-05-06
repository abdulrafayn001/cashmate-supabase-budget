// _shared/mcmc/linalg.ts
// ============================================================================
// Linear algebra primitives on Float64Array, kept allocation-frugal so they
// can sit inside MCMC inner loops without churning the GC.
//
// Matrices are row-major. A d×d matrix lives in a Float64Array of length d*d
// where M[i,j] = arr[i*d + j].
// ============================================================================

/** Cholesky decomposition: A = L · L^T (L lower-triangular).
 *  Writes L in-place into `out` (which may equal `A`); strictly-upper entries
 *  are zeroed. Returns true on success, false if A is not PSD. */
export function cholesky(A: Float64Array, d: number, out?: Float64Array): boolean {
  const L = out ?? new Float64Array(d * d)
  if (L !== A) L.set(A)
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let s = L[i * d + j]
      for (let k = 0; k < j; k++) s -= L[i * d + k] * L[j * d + k]
      if (i === j) {
        if (s <= 0) return false
        L[i * d + j] = Math.sqrt(s)
      } else {
        L[i * d + j] = s / L[j * d + j]
      }
    }
    // Zero the strictly upper triangle so callers can MVN-sample without care.
    for (let j = i + 1; j < d; j++) L[i * d + j] = 0
  }
  return true
}

/** Forward substitution: solve L · x = b for lower-triangular L.
 *  Writes the solution into `out` (may equal `b`). */
export function solveLowerTri(
  L: Float64Array,
  b: Float64Array,
  d: number,
  out?: Float64Array,
): Float64Array {
  const x = out ?? new Float64Array(d)
  for (let i = 0; i < d; i++) {
    let s = b[i]
    for (let j = 0; j < i; j++) s -= L[i * d + j] * x[j]
    x[i] = s / L[i * d + i]
  }
  return x
}

/** Back substitution: solve L^T · x = b for lower-triangular L. */
export function solveLowerTriTransposed(
  L: Float64Array,
  b: Float64Array,
  d: number,
  out?: Float64Array,
): Float64Array {
  const x = out ?? new Float64Array(d)
  for (let i = d - 1; i >= 0; i--) {
    let s = b[i]
    for (let j = i + 1; j < d; j++) s -= L[j * d + i] * x[j]
    x[i] = s / L[i * d + i]
  }
  return x
}

/** Solve A · x = b given A's Cholesky L (lower).
 *  Two triangular solves: forward then backward. */
export function choleskySolve(
  L: Float64Array,
  b: Float64Array,
  d: number,
  out?: Float64Array,
): Float64Array {
  const y = solveLowerTri(L, b, d)
  return solveLowerTriTransposed(L, y, d, out)
}

/** y = A · x. A is d×d row-major, x is length-d. */
export function matvec(
  A: Float64Array,
  x: Float64Array,
  d: number,
  out?: Float64Array,
): Float64Array {
  const y = out ?? new Float64Array(d)
  for (let i = 0; i < d; i++) {
    let s = 0
    for (let j = 0; j < d; j++) s += A[i * d + j] * x[j]
    y[i] = s
  }
  return y
}

/** C = A · B (all d×d row-major). */
export function matmul(
  A: Float64Array,
  B: Float64Array,
  d: number,
  out?: Float64Array,
): Float64Array {
  const C = out ?? new Float64Array(d * d)
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      let s = 0
      for (let k = 0; k < d; k++) s += A[i * d + k] * B[k * d + j]
      C[i * d + j] = s
    }
  }
  return C
}

/** Build a symmetric d×d matrix from a flat correlation matrix R and a stdev
 *  vector sigma: Σ = diag(σ) · R · diag(σ). */
export function covarianceFromCorr(
  sigma: Float64Array,
  R: Float64Array,
  d: number,
  out?: Float64Array,
): Float64Array {
  const Sigma = out ?? new Float64Array(d * d)
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      Sigma[i * d + j] = sigma[i] * R[i * d + j] * sigma[j]
    }
  }
  return Sigma
}

// ---------------------------------------------------------------------------
// Higham (2002) nearest-PSD projection — a single Newton-style sweep that
// pushes negative eigenvalues to zero. Used as a fallback when a user view
// override breaks the prior covariance's PSDness. Phase 1 ships a simple
// jitter-then-cholesky retry; the full Higham sweep arrives in Phase 4.
// ---------------------------------------------------------------------------

/** Add jitter * I until Cholesky succeeds. Returns the L on success. */
export function choleskyWithJitter(
  A: Float64Array,
  d: number,
  jitter0 = 1e-10,
  maxAttempts = 8,
): { L: Float64Array; jitter: number } | null {
  const work = new Float64Array(A.length)
  let jitter = 0
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    work.set(A)
    if (jitter > 0) {
      for (let i = 0; i < d; i++) work[i * d + i] += jitter
    }
    const L = new Float64Array(d * d)
    if (cholesky(work, d, L)) return { L, jitter }
    jitter = jitter === 0 ? jitter0 : jitter * 10
  }
  return null
}
