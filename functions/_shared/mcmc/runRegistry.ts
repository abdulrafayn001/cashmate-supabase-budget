// _shared/mcmc/runRegistry.ts
// ============================================================================
// Helpers around the public.mcmc_runs table.
// All writes go through a Supabase client passed in by the caller — RLS is
// already enforced server-side. The helper does NOT trust the caller's
// user_id; it expects the caller to pass a JWT-derived id and the SQL row
// policy `user_id = auth.uid()` to fail-safe on mismatch.
//
// Cache key: sha256(user_id || feature || canonical_json(params) || data_fp).
// Phase 1 ships only createRun / finalizeRun / failRun — checkpoint resume
// arrives with A1 in Phase 2.
// ============================================================================

import type { Feature, FinalizedRun } from './types.ts'

// We type the Supabase client structurally so this file has zero runtime
// imports (and therefore zero TS errors when run under Deno without the SDK
// installed, e.g. during unit tests).
export interface SupabaseLike {
  from(table: string): {
    insert(row: Record<string, unknown>): {
      select(): { single(): Promise<{ data: { id: string } | null; error: unknown }> }
    }
    update(row: Record<string, unknown>): {
      eq(col: string, val: string): Promise<{ error: unknown }>
    }
    delete(): {
      eq(col: string, val: string): {
        eq(col: string, val: string): {
          eq(col: string, val: string): {
            eq(col: string, val: string): Promise<{ error: unknown }>
          }
        }
      }
    }
    select(cols: string): {
      eq(col: string, val: string): {
        eq(col: string, val: string): {
          eq(col: string, val: string): {
            gt(col: string, val: string): {
              order(col: string, opts: { ascending: boolean }): {
                limit(n: number): Promise<{
                  data: Array<{ id: string; samples_summary_json: unknown }> | null
                  error: unknown
                }>
              }
            }
          }
        }
      }
    }
  }
}

export interface CreateRunArgs {
  userId: string
  feature: Feature
  paramsJson: Record<string, unknown>
  cacheKey: string
  dataFingerprint?: string | null
  nChains?: number
  nIter?: number
}

/** Insert a 'queued' row, return its id. Caller is responsible for moving
 *  the row to 'running' / 'complete' / 'failed' as the work progresses. */
export async function createRun(
  supabase: SupabaseLike,
  args: CreateRunArgs,
): Promise<string> {
  const { data, error } = await supabase
    .from('mcmc_runs')
    .insert({
      user_id: args.userId,
      feature: args.feature,
      status: 'running',
      params_json: args.paramsJson,
      cache_key: args.cacheKey,
      data_fingerprint: args.dataFingerprint ?? null,
      n_chains: args.nChains ?? 4,
      n_iter: args.nIter ?? null,
    })
    .select()
    .single()

  if (error || !data) {
    throw new Error(`createRun: insert failed (${stringifyError(error)})`)
  }
  return data.id
}

/** Mark a run complete, write summary + diagnostics.
 *
 *  If `cacheInvalidation` is provided, any older `status='complete'` rows
 *  matching the same (user_id, feature, cache_key) are deleted *before*
 *  this row's status flips. This is required because the partial unique
 *  index `mcmc_runs_cache_idx ... WHERE status='complete'` allows only one
 *  complete row per cache key — without this step, force-recompute or a
 *  stale-cache path would hit a duplicate-key violation. */
export async function finalizeRun(
  supabase: SupabaseLike,
  runId: string,
  result: FinalizedRun,
  cacheInvalidation?: { userId: string; feature: Feature; cacheKey: string },
): Promise<void> {
  if (cacheInvalidation) {
    const { error: delErr } = await supabase
      .from('mcmc_runs')
      .delete()
      .eq('user_id', cacheInvalidation.userId)
      .eq('feature', cacheInvalidation.feature)
      .eq('cache_key', cacheInvalidation.cacheKey)
      .eq('status', 'complete')
    if (delErr) {
      throw new Error(`finalizeRun: invalidate failed (${stringifyError(delErr)})`)
    }
  }

  const { error } = await supabase
    .from('mcmc_runs')
    .update({
      status: 'complete',
      n_samples: result.nSamples,
      runtime_ms: result.runtimeMs,
      rhat_max: result.diagnostics.rhatMax,
      ess_min: result.diagnostics.essMin,
      diagnostics_pass: result.diagnostics.pass,
      samples_summary_json: result.summary,
    })
    .eq('id', runId)

  if (error) throw new Error(`finalizeRun: update failed (${stringifyError(error)})`)
}

/** Mark a run failed with a short message (truncated to keep the row sane). */
export async function failRun(
  supabase: SupabaseLike,
  runId: string,
  message: string,
): Promise<void> {
  const trimmed = message.length > 2000 ? message.slice(0, 2000) + '…' : message
  const { error } = await supabase
    .from('mcmc_runs')
    .update({ status: 'failed', error_message: trimmed })
    .eq('id', runId)

  if (error) throw new Error(`failRun: update failed (${stringifyError(error)})`)
}

/** Look up a recent complete run for cache reuse.
 *  Returns the most-recent matching row's summary, or null if none. */
export async function findCachedRun(
  supabase: SupabaseLike,
  userId: string,
  feature: Feature,
  cacheKey: string,
  maxAgeHours = 24,
): Promise<{ id: string; summary: unknown } | null> {
  const cutoff = new Date(Date.now() - maxAgeHours * 3600_000).toISOString()
  const { data, error } = await supabase
    .from('mcmc_runs')
    .select('id, samples_summary_json')
    .eq('user_id', userId)
    .eq('feature', feature)
    .eq('cache_key', cacheKey)
    .gt('created_at', cutoff)
    .order('created_at', { ascending: false })
    .limit(1)

  if (error) throw new Error(`findCachedRun: select failed (${stringifyError(error)})`)
  if (!data || data.length === 0) return null
  return { id: data[0].id, summary: data[0].samples_summary_json }
}

// ---------------------------------------------------------------------------
// Cache key — sha256 of canonical inputs. Uses Web Crypto, available in Deno
// edge runtime and modern browsers.
// ---------------------------------------------------------------------------
export async function buildCacheKey(
  userId: string,
  feature: Feature,
  params: Record<string, unknown>,
  dataFingerprint: string | null,
): Promise<string> {
  const canonical = canonicalJson(params)
  const input = `${userId}|${feature}|${canonical}|${dataFingerprint ?? ''}`
  const buf = new TextEncoder().encode(input)
  const hash = await crypto.subtle.digest('SHA-256', buf)
  return hex(new Uint8Array(hash))
}

/** Canonicalize JSON so the same params object always hashes the same way:
 *  recursively sort object keys, leave arrays alone (order is meaningful). */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value)
  if (Array.isArray(value)) return '[' + value.map(canonicalJson).join(',') + ']'
  const keys = Object.keys(value as Record<string, unknown>).sort()
  const parts = keys.map(
    (k) => JSON.stringify(k) + ':' + canonicalJson((value as Record<string, unknown>)[k]),
  )
  return '{' + parts.join(',') + '}'
}

function hex(bytes: Uint8Array): string {
  let s = ''
  for (let i = 0; i < bytes.length; i++) {
    s += bytes[i].toString(16).padStart(2, '0')
  }
  return s
}

function stringifyError(e: unknown): string {
  if (e === null || e === undefined) return 'unknown'
  if (typeof e === 'string') return e
  if (typeof e === 'object' && 'message' in e) return String((e as { message: unknown }).message)
  try {
    return JSON.stringify(e)
  } catch {
    return String(e)
  }
}
