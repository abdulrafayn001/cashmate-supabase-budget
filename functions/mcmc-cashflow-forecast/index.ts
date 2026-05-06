// Supabase Edge Function: mcmc-cashflow-forecast
// ============================================================================
// A1 — Probabilistic cash-flow & net-worth forecast (CashMate MCMC suite §2).
//
// Request  POST /functions/v1/mcmc-cashflow-forecast  Authorization: Bearer <jwt>
//   {
//     "params": {
//       "horizon": 6,             // months ahead, default 6
//       "monthsHistory": 24,      // months of transactions to aggregate, default 24
//     },
//     "options": {
//       "n_chains": 4,            // default 4
//       "n_iter": 2000,           // default 2000
//       "seed": "user-x:run-y",   // optional explicit seed; default derived
//       "force_recompute": false, // ignore cache
//       "async": false            // 202 + EdgeRuntime.waitUntil() if true
//     }
//   }
//
// Response 200 (sync)
//   { run_id, status, n_samples, runtime_ms, diagnostics, summary, forecast }
// Response 202 (async)
//   { run_id, status: "queued", realtime_channel: "mcmc_runs:id=eq.<id>" }
//
// Uses the user's JWT for all DB access — RLS scopes transactions + mcmc_runs
// rows to the authenticated user. The shared MCMC library is dependency-free
// so this file is the only place a Supabase import lives.
// ============================================================================

import { createClient } from 'jsr:@supabase/supabase-js@2'
import {
  aggregateMonthly,
  runCashflow,
  type CashflowResult,
  type TxRow,
} from '../_shared/mcmc/features/cashflow.ts'
import {
  buildCacheKey,
  createRun,
  failRun,
  finalizeRun,
  findCachedRun,
  type SupabaseLike,
} from '../_shared/mcmc/runRegistry.ts'

interface RequestBody {
  params?: {
    horizon?: number
    monthsHistory?: number
  }
  options?: {
    n_chains?: number
    n_iter?: number
    seed?: string
    force_recompute?: boolean
    async?: boolean
  }
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

Deno.serve(async (req: Request) => {
  if (req.method !== 'POST') return json(405, { error: 'Method not allowed' })

  const auth = req.headers.get('Authorization')
  if (!auth) return json(401, { error: 'Missing Authorization header' })

  let body: RequestBody = {}
  try {
    body = (await req.json()) as RequestBody
  } catch {
    // Empty body is fine — defaults all the way down.
  }

  const horizon = clampInt(body.params?.horizon ?? 6, 1, 12)
  const monthsHistory = clampInt(body.params?.monthsHistory ?? 24, 6, 60)
  const nChains = clampInt(body.options?.n_chains ?? 4, 1, 8)
  const nIter = clampInt(body.options?.n_iter ?? 2000, 200, 10_000)
  const forceRecompute = !!body.options?.force_recompute
  const isAsync = !!body.options?.async

  const SUPABASE_URL = Deno.env.get('SUPABASE_URL')
  const SUPABASE_ANON_KEY = Deno.env.get('SUPABASE_ANON_KEY')
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
    return json(500, { error: 'Missing Supabase env' })
  }

  // Auth-scoped client — RLS will reject anything not owned by the JWT user.
  const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: auth } },
  })

  const { data: userRes, error: userErr } = await supabase.auth.getUser()
  if (userErr || !userRes?.user) return json(401, { error: 'Invalid JWT' })
  const userId = userRes.user.id

  // -------------------------------------------------------------------------
  // 1. Pull transactions for the requested window (RLS already scopes by user)
  // -------------------------------------------------------------------------
  const since = new Date()
  since.setUTCMonth(since.getUTCMonth() - monthsHistory)
  since.setUTCDate(1)
  since.setUTCHours(0, 0, 0, 0)

  const { data: txData, error: txErr } = await supabase
    .from('transactions')
    .select('date, type, base_currency_amount, source_amount')
    .gte('date', since.toISOString())
    .in('type', ['credit', 'debit'])
    .order('date', { ascending: true })

  if (txErr) return json(500, { error: 'transactions query failed', detail: String(txErr) })

  type TxRowRaw = {
    date: string
    type: string
    base_currency_amount: number | null
    source_amount: number | null
  }
  const rawRows = (txData ?? []) as TxRowRaw[]

  const rows: TxRow[] = rawRows
    .map((r) => ({
      date: r.date,
      type: r.type,
      amount: r.base_currency_amount ?? r.source_amount ?? 0,
    }))
    .filter((r) => r.amount > 0)

  const data = aggregateMonthly(rows)
  if (data.c.length < 6) {
    return json(400, {
      error: 'Insufficient data',
      detail: `Need ≥ 6 months with both income and spend; got ${data.c.length}`,
    })
  }

  // -------------------------------------------------------------------------
  // 2. Build cache key. Fingerprint = #rows || lastDate (cheap and stable).
  // -------------------------------------------------------------------------
  const lastDate = rawRows.length > 0 ? rawRows[rawRows.length - 1].date : ''
  const dataFingerprint = `${rawRows.length}|${lastDate}|${monthsHistory}m`
  const params = { horizon, monthsHistory, nChains, nIter }
  const cacheKey = await buildCacheKey(userId, 'cashflow_forecast', params, dataFingerprint)

  // -------------------------------------------------------------------------
  // 3. Cache lookup (unless force_recompute)
  // -------------------------------------------------------------------------
  if (!forceRecompute) {
    const cached = await findCachedRun(
      supabase as unknown as SupabaseLike,
      userId,
      'cashflow_forecast',
      cacheKey,
      24,
    )
    if (cached) {
      return json(200, {
        run_id: cached.id,
        status: 'complete',
        cached: true,
        result: cached.summary,
      })
    }
  }

  // -------------------------------------------------------------------------
  // 4. Create the run row → run sampler → finalize
  // -------------------------------------------------------------------------
  // Deterministic seed: same data + same params → identical forecast.
  // Without this, every force-refresh produces different bands due to MCMC
  // Monte-Carlo error, even when nothing about the user's data has changed.
  const seed = body.options?.seed ?? `${userId}:cashflow:${cacheKey}`
  const runId = await createRun(supabase as unknown as SupabaseLike, {
    userId,
    feature: 'cashflow_forecast',
    paramsJson: { ...params, dataMonths: data.months.length },
    cacheKey,
    dataFingerprint,
    nChains,
    nIter,
  })

  const sampleAndStore = async () => {
    try {
      const result = await runCashflow(data, {
        seed,
        nIter,
        nBurn: Math.floor(nIter / 2),
        thin: 2,
        nChains,
        horizon,
        yieldFn: () => new Promise<void>((r) => setTimeout(r, 0)),
      })
      await finalizeRun(
        supabase as unknown as SupabaseLike,
        runId,
        {
          nSamples: result.nSamples,
          runtimeMs: result.runtimeMs,
          diagnostics: {
            rhatMax: result.diagnostics.rhatMax,
            essMin: result.diagnostics.essMin,
            nDivergences: 0,
            pass: result.diagnostics.pass,
          },
          summary: shapeSummary(result),
        },
        { userId, feature: 'cashflow_forecast', cacheKey },
      )
      return result
    } catch (e) {
      await failRun(supabase as unknown as SupabaseLike, runId, String(e))
      throw e
    }
  }

  // -------------------------------------------------------------------------
  // 5. Sync vs async dispatch
  // -------------------------------------------------------------------------
  if (isAsync) {
    // @ts-expect-error — EdgeRuntime is a Deno deploy global; not in lib.
    EdgeRuntime.waitUntil(sampleAndStore())
    return json(202, {
      run_id: runId,
      status: 'queued',
      realtime_channel: `mcmc_runs:id=eq.${runId}`,
    })
  }

  try {
    const result = await sampleAndStore()
    return json(200, {
      run_id: runId,
      status: 'complete',
      n_samples: result.nSamples,
      runtime_ms: result.runtimeMs,
      diagnostics: result.diagnostics,
      summary: result.summary,
      forecast: shapeForecast(result),
    })
  } catch (e) {
    return json(500, { run_id: runId, error: String(e) })
  }
})

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function clampInt(x: unknown, min: number, max: number): number {
  const n = typeof x === 'number' && Number.isFinite(x) ? Math.floor(x) : min
  return Math.min(max, Math.max(min, n))
}

/** Trim & flatten the result for JSON storage in samples_summary_json. */
function shapeSummary(result: CashflowResult): Record<string, unknown> {
  return {
    diagnostics: result.diagnostics,
    summary: result.summary,
    forecast: shapeForecast(result),
    nSamples: result.nSamples,
    runtimeMs: result.runtimeMs,
  }
}

function shapeForecast(result: CashflowResult): Record<string, unknown> {
  const f = result.forecast
  return {
    months: f.forecastMonths,
    income: floatArrayMap(f.income),
    spend: floatArrayMap(f.spend),
    net: floatArrayMap(f.net),
    cumulativeNet: floatArrayMap(f.cumulativeNet),
  }
}

function floatArrayMap(s: { median: Float64Array; q05: Float64Array; q25: Float64Array; q75: Float64Array; q95: Float64Array }) {
  return {
    median: Array.from(s.median),
    q05: Array.from(s.q05),
    q25: Array.from(s.q25),
    q75: Array.from(s.q75),
    q95: Array.from(s.q95),
  }
}
