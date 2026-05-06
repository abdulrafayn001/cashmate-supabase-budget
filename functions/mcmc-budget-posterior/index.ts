// Supabase Edge Function: mcmc-budget-posterior
// ============================================================================
// A2 — Bayesian budget posterior (CashMate MCMC suite §2 A2).
//
// Replaces category_budgets.spent_amount (a single number) with:
//   - per-category posterior on monthly mean spend (PKR)
//   - per-category posterior on the active period's total = spent_so_far + remainder
//   - per-category P(period total > allocated)
//   - per-group rollups of the same
//
// Request  POST /functions/v1/mcmc-budget-posterior  Authorization: Bearer <jwt>
//   {
//     "params": {
//       "masterBudgetId": "uuid",     // optional; defaults to active is_active=true row
//       "monthsHistory": 12           // months of debit history, clamp 6..24, default 12
//     },
//     "options": {
//       "n_chains": 4, "n_iter": 3000,
//       "seed": "...", "force_recompute": false, "async": false
//     }
//   }
//
// Response 200 (sync): { run_id, status, n_samples, runtime_ms, diagnostics,
//                        summary, posteriors, group_totals, period }
// Response 202 (async): { run_id, status: "queued", realtime_channel }
//
// Auth-scoped Supabase client → RLS scopes transactions/budgets/categories
// to the authenticated user. The shared MCMC library is dependency-free so
// this file is the only place a Supabase import lives.
// ============================================================================

import { createClient } from 'jsr:@supabase/supabase-js@2'
import {
  aggregateMonthly,
  type BudgetPeriodMeta,
  type BudgetResult,
  type CategoryMeta,
  type ParentGroup,
  runBudget,
  type TxRow,
} from '../_shared/mcmc/features/budget.ts'
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
    masterBudgetId?: string
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

const EXPENSE_GROUPS: ReadonlySet<string> = new Set(['needs', 'wants', 'savings'])

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function clampInt(x: unknown, min: number, max: number): number {
  const n = typeof x === 'number' && Number.isFinite(x) ? Math.floor(x) : min
  return Math.min(max, Math.max(min, n))
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

  const monthsHistory = clampInt(body.params?.monthsHistory ?? 12, 6, 24)
  const nChains = clampInt(body.options?.n_chains ?? 4, 1, 8)
  const nIter = clampInt(body.options?.n_iter ?? 3000, 200, 10_000)
  const forceRecompute = !!body.options?.force_recompute
  const isAsync = !!body.options?.async

  const SUPABASE_URL = Deno.env.get('SUPABASE_URL')
  const SUPABASE_ANON_KEY = Deno.env.get('SUPABASE_ANON_KEY')
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
    return json(500, { error: 'Missing Supabase env' })
  }

  const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: auth } },
  })

  const { data: userRes, error: userErr } = await supabase.auth.getUser()
  if (userErr || !userRes?.user) return json(401, { error: 'Invalid JWT' })
  const userId = userRes.user.id

  // -------------------------------------------------------------------------
  // 1. Resolve master_budget. Either use the provided id or the active row.
  // -------------------------------------------------------------------------
  let masterBudgetId = body.params?.masterBudgetId
  let mbRow: { id: string; start_date: string; end_date: string; is_active: boolean } | null = null
  if (masterBudgetId) {
    const { data, error } = await supabase
      .from('master_budgets')
      .select('id, start_date, end_date, is_active')
      .eq('id', masterBudgetId)
      .single()
    if (error || !data) return json(404, { error: 'master_budget not found' })
    mbRow = data
  } else {
    const { data, error } = await supabase
      .from('master_budgets')
      .select('id, start_date, end_date, is_active')
      .eq('is_active', true)
      .order('start_date', { ascending: false })
      .limit(1)
      .maybeSingle()
    if (error) return json(500, { error: 'master_budgets query failed', detail: String(error) })
    if (!data) {
      return json(404, {
        error: 'No active master budget',
        detail: 'Create a budget first, or pass params.masterBudgetId explicitly.',
      })
    }
    mbRow = data
    masterBudgetId = data.id
  }

  // -------------------------------------------------------------------------
  // 2. Pull category_budgets for this period — these are the categories the
  //    user wants A2 to fit. Drop non-expense rows (income / unbudgeted).
  // -------------------------------------------------------------------------
  const { data: cbData, error: cbErr } = await supabase
    .from('category_budgets')
    .select('category_id, allocated_amount, parent_group')
    .eq('master_budget_id', masterBudgetId)
  if (cbErr) return json(500, { error: 'category_budgets query failed', detail: String(cbErr) })

  type CB = { category_id: string; allocated_amount: number; parent_group: string }
  const cbRows = ((cbData ?? []) as CB[]).filter((r) => EXPENSE_GROUPS.has(r.parent_group))
  if (cbRows.length === 0) {
    return json(400, {
      error: 'No expense categories budgeted',
      detail: 'A2 fits only needs/wants/savings categories with allocated_amount > 0.',
    })
  }

  const categoryIds = cbRows.map((r) => r.category_id)

  // -------------------------------------------------------------------------
  // 3. Pull the category metadata (names, parent_category — confirm match).
  // -------------------------------------------------------------------------
  const { data: catData, error: catErr } = await supabase
    .from('categories')
    .select('id, name, parent_category')
    .in('id', categoryIds)
  if (catErr) return json(500, { error: 'categories query failed', detail: String(catErr) })

  type Cat = { id: string; name: string; parent_category: string }
  const catRows = (catData ?? []) as Cat[]
  const catById = new Map<string, Cat>(catRows.map((c) => [c.id, c]))

  const categories: CategoryMeta[] = []
  for (const cb of cbRows) {
    const cat = catById.get(cb.category_id)
    if (!cat) continue
    if (!EXPENSE_GROUPS.has(cat.parent_category)) continue
    categories.push({
      id: cb.category_id,
      name: cat.name,
      parentGroup: cat.parent_category as ParentGroup,
      allocated: Number(cb.allocated_amount) || 0,
    })
  }
  if (categories.length === 0) {
    return json(400, { error: 'No matching expense categories', detail: 'category_budgets present but categories not found / not expense-side.' })
  }

  // -------------------------------------------------------------------------
  // 4. Pull debit transactions for the history window.
  // -------------------------------------------------------------------------
  const since = new Date()
  since.setUTCMonth(since.getUTCMonth() - monthsHistory)
  since.setUTCDate(1)
  since.setUTCHours(0, 0, 0, 0)

  const { data: txData, error: txErr } = await supabase
    .from('transactions')
    .select('date, category_id, type, base_currency_amount, source_amount, exclude_from_budget')
    .gte('date', since.toISOString())
    .eq('type', 'debit')
    .in('category_id', categoryIds)
    .order('date', { ascending: true })
  if (txErr) return json(500, { error: 'transactions query failed', detail: String(txErr) })

  type TxRaw = {
    date: string
    category_id: string | null
    type: string
    base_currency_amount: number | null
    source_amount: number | null
    exclude_from_budget: boolean | null
  }
  const rawRows = (txData ?? []) as TxRaw[]
  const rows: TxRow[] = []
  for (const r of rawRows) {
    if (!r.category_id) continue
    if (r.exclude_from_budget) continue
    const amount = Number(r.base_currency_amount ?? r.source_amount ?? 0)
    if (!isFinite(amount) || amount <= 0) continue
    rows.push({ date: r.date, categoryId: r.category_id, amount })
  }

  const data = aggregateMonthly(rows, categories)
  // Insufficient-data threshold: ≥ 3 distinct months OR ≥ 6 total observations.
  const totalObs = (() => {
    let s = 0
    for (let c = 0; c < data.nObs.length; c++) s += data.nObs[c]
    return s
  })()
  if (data.months.length < 3 || totalObs < 6) {
    return json(400, {
      error: 'Insufficient data',
      detail: `Need ≥ 3 distinct months and ≥ 6 categorised debit transactions; got ${data.months.length} months, ${totalObs} transactions.`,
    })
  }

  // -------------------------------------------------------------------------
  // 5. Compute spent-so-far per category for the active period.
  // -------------------------------------------------------------------------
  const spentSoFar: Record<string, number> = {}
  for (const c of categories) spentSoFar[c.id] = 0
  const periodStartIso = mbRow.start_date
  const periodStartMs = new Date(periodStartIso).getTime()
  const periodEndIso = mbRow.end_date
  const todayIso = new Date().toISOString().slice(0, 10)
  const todayMs = Date.now()
  for (const r of rows) {
    const ts = new Date(r.date).getTime()
    if (ts >= periodStartMs && ts <= todayMs) {
      spentSoFar[r.categoryId] = (spentSoFar[r.categoryId] ?? 0) + r.amount
    }
  }

  const periodMeta: BudgetPeriodMeta = {
    startDate: periodStartIso,
    endDate: periodEndIso,
    today: todayIso,
    spentSoFar,
  }

  // -------------------------------------------------------------------------
  // 6. Cache key: data fingerprint includes row count, last-tx date, period id,
  //    today, and history window. Today is included so re-runs after midnight
  //    re-forecast a shorter remainder; same data + same day = same result.
  // -------------------------------------------------------------------------
  const lastDate = rawRows.length > 0 ? rawRows[rawRows.length - 1].date : ''
  const dataFingerprint = `${rawRows.length}|${lastDate}|${masterBudgetId}|${monthsHistory}m|${todayIso}`
  const params = { masterBudgetId, monthsHistory, nChains, nIter }
  const cacheKey = await buildCacheKey(userId, 'budget_posterior', params, dataFingerprint)

  if (!forceRecompute) {
    const cached = await findCachedRun(
      supabase as unknown as SupabaseLike,
      userId,
      'budget_posterior',
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
  // 7. Create run, sample, finalize.
  // -------------------------------------------------------------------------
  const seed = body.options?.seed ?? `${userId}:budget:${cacheKey}`
  const runId = await createRun(supabase as unknown as SupabaseLike, {
    userId,
    feature: 'budget_posterior',
    paramsJson: { ...params, dataMonths: data.months.length, nCategories: categories.length },
    cacheKey,
    dataFingerprint,
    nChains,
    nIter,
  })

  const sampleAndStore = async () => {
    try {
      const result = await runBudget(data, periodMeta, {
        seed,
        nIter,
        nBurn: Math.floor(nIter / 2),
        thin: 2,
        nChains,
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
        { userId, feature: 'budget_posterior', cacheKey },
      )
      return result
    } catch (e) {
      await failRun(supabase as unknown as SupabaseLike, runId, String(e))
      throw e
    }
  }

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
      posteriors: result.posteriors,
      group_totals: result.groupTotals,
      period: result.period,
    })
  } catch (e) {
    return json(500, { run_id: runId, error: String(e) })
  }
})

/** Trim & flatten the result for JSON storage in samples_summary_json.
 *  Mirrors A1's shapeSummary so the BFF can read either response shape. */
function shapeSummary(result: BudgetResult): Record<string, unknown> {
  return {
    diagnostics: result.diagnostics,
    summary: result.summary,
    posteriors: result.posteriors,
    group_totals: result.groupTotals,
    period: result.period,
    nSamples: result.nSamples,
    runtimeMs: result.runtimeMs,
  }
}
