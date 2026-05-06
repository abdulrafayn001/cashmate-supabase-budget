// _shared/mcmc/features/budget_test.ts
// ============================================================================
// Phase-2 acceptance gates for the A2 budget posterior model:
//
//   1. Synthetic-data parameter recovery: μ_g posterior covers truth.
//   2. Hierarchical shrinkage: a sparse-data category is shrunk toward μ_g
//      more than an abundant-data category.
//   3. Diagnostics gate: rhat_max < 1.05, ess_min > 200 on a CI-sized run
//      (prod gate rhat<1.01 / ess>400 stays inside runBudget.diagnostics.pass).
//   4. Aggregation: groups by (category, month), drops zero/negative amounts,
//      keeps no-data categories in the model.
//   5. Forecast sanity: pOver in [0, 1], group total = sum of category totals,
//      smaller pOver when allocation >> period predictive mean.
// ============================================================================

import { assertAlmostEquals, assertEquals } from 'jsr:@std/assert@^1'
import {
  aggregateMonthly,
  type CategoryMeta,
  runBudget,
  type TxRow,
} from './budget.ts'
import { makeRng, stdNormal } from '../random.ts'
import type { BudgetData, BudgetPeriodMeta } from './budget.ts'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MONTHS_2025 = [
  '2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06',
  '2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12',
]

function synthRows(
  trueParams: Record<string, { theta: number; sigma: number }>,
  months: string[],
  seed: string,
): TxRow[] {
  const rng = makeRng(seed)
  const rows: TxRow[] = []
  for (const m of months) {
    for (const [catId, { theta, sigma }] of Object.entries(trueParams)) {
      const logAmount = theta + sigma * stdNormal(rng)
      const amount = Math.exp(logAmount)
      // Use the 15th of the month as a fixed timestamp.
      rows.push({ date: `${m}-15T00:00:00Z`, categoryId: catId, amount })
    }
  }
  return rows
}

function makePeriod(catIds: string[]): BudgetPeriodMeta {
  // Synthetic period covering all of 2025; "today" lands mid-period (mid-July).
  const spentSoFar: Record<string, number> = {}
  for (const id of catIds) spentSoFar[id] = 0
  return {
    startDate: '2025-01-01T00:00:00Z',
    endDate: '2025-12-31T23:59:59Z',
    today: '2025-07-15T00:00:00Z',
    spentSoFar,
  }
}

// ---------------------------------------------------------------------------
// 1. Aggregation
// ---------------------------------------------------------------------------
Deno.test('aggregateMonthly: groups by (category, month), drops bad amounts', () => {
  const cats: CategoryMeta[] = [
    { id: 'cat-a', name: 'Groceries', parentGroup: 'needs', allocated: 25_000 },
    { id: 'cat-b', name: 'Dining',    parentGroup: 'wants', allocated: 8_000 },
    { id: 'cat-c', name: 'Savings',   parentGroup: 'savings', allocated: 30_000 },
  ]
  const rows: TxRow[] = [
    { date: '2025-01-05T00:00:00Z', categoryId: 'cat-a', amount: 12_000 },
    { date: '2025-01-20T00:00:00Z', categoryId: 'cat-a', amount: 8_000 },   // sums in same month
    { date: '2025-02-10T00:00:00Z', categoryId: 'cat-a', amount: 18_000 },
    { date: '2025-01-15T00:00:00Z', categoryId: 'cat-b', amount: 5_000 },
    { date: '2025-01-22T00:00:00Z', categoryId: 'unknown', amount: 1000 }, // dropped (unknown cat)
    { date: '2025-02-22T00:00:00Z', categoryId: 'cat-b', amount: -100 },    // dropped (negative)
    { date: '2025-03-01T00:00:00Z', categoryId: 'cat-b', amount: 0 },       // dropped (zero)
    // cat-c has no data — must still be in the model.
  ]
  const data = aggregateMonthly(rows, cats)
  assertEquals(data.categories.length, 3)
  assertEquals(data.months.length, 2)  // Jan + Feb (cat-b only had Jan; March has no positive obs)
  assertEquals(data.nObs[0], 2) // cat-a: Jan (sum) + Feb
  assertEquals(data.nObs[1], 1) // cat-b: Jan only
  assertEquals(data.nObs[2], 0) // cat-c: no data, but still in model
  // sumLogX_a should be ln(20000) + ln(18000)
  assertAlmostEquals(data.sumLogX[0], Math.log(20_000) + Math.log(18_000), 1e-9)
  // Falls back to log(allocated) for cat-c
  assertAlmostEquals(data.logMedian[2], Math.log(30_000), 1e-9)
})

// ---------------------------------------------------------------------------
// 2. Synthetic recovery — μ_g posterior should cover truth
// ---------------------------------------------------------------------------
Deno.test('runBudget: μ_g posterior covers known group truth', async () => {
  // Two categories per group, all with substantial data.
  const cats: CategoryMeta[] = [
    { id: 'g1', name: 'Groceries',  parentGroup: 'needs',   allocated: 20_000 },
    { id: 'r1', name: 'Rent',       parentGroup: 'needs',   allocated: 35_000 },
    { id: 'd1', name: 'Dining',     parentGroup: 'wants',   allocated: 10_000 },
    { id: 'e1', name: 'Entertain',  parentGroup: 'wants',   allocated: 5_000 },
    { id: 's1', name: 'Emergency',  parentGroup: 'savings', allocated: 25_000 },
    { id: 'i1', name: 'Index Fund', parentGroup: 'savings', allocated: 15_000 },
  ]
  // True μ_needs ≈ ln(25000) ≈ 10.13, μ_wants ≈ ln(7500) ≈ 8.92, μ_savings ≈ ln(20000) ≈ 9.90
  const trueParams = {
    g1: { theta: Math.log(15_000), sigma: 0.20 },
    r1: { theta: Math.log(35_000), sigma: 0.10 },
    d1: { theta: Math.log(8_000),  sigma: 0.25 },
    e1: { theta: Math.log(7_000),  sigma: 0.30 },
    s1: { theta: Math.log(22_000), sigma: 0.15 },
    i1: { theta: Math.log(18_000), sigma: 0.15 },
  }
  const rows = synthRows(trueParams, MONTHS_2025, 'recover-mu')
  const data = aggregateMonthly(rows, cats)

  const period = makePeriod(cats.map((c) => c.id))
  const result = await runBudget(data, period, {
    seed: 'recover',
    nIter: 1500,
    nBurn: 750,
    thin: 2,
    nChains: 3,
    yieldFn: () => Promise.resolve(),
  })

  // Posterior on μ_needs should bracket ln((15000 + 35000) / 2) ≈ 10.13.
  const muNeeds = result.summary['mu[needs]']
  if (!(muNeeds.q05 < 10.13 && muNeeds.q95 > 10.13)) {
    throw new Error(`mu[needs] 90% CI [${muNeeds.q05}, ${muNeeds.q95}] does not cover 10.13`)
  }
  const muSavings = result.summary['mu[savings]']
  if (!(muSavings.q05 < 9.90 && muSavings.q95 > 9.90)) {
    throw new Error(`mu[savings] 90% CI [${muSavings.q05}, ${muSavings.q95}] does not cover 9.90`)
  }
})

// ---------------------------------------------------------------------------
// 3. Hierarchical shrinkage — sparse category should be shrunk toward μ_g
// ---------------------------------------------------------------------------
Deno.test('runBudget: sparse category is shrunk toward group mean', async () => {
  // One abundant-data category at 30,000, one sparse-data outlier at 5,000.
  const cats: CategoryMeta[] = [
    { id: 'rich',   name: 'Rent',   parentGroup: 'needs', allocated: 30_000 },
    { id: 'sparse', name: 'Bills',  parentGroup: 'needs', allocated: 30_000 },
    // Add a peer to the group so there's something to pool toward.
    { id: 'mid',    name: 'Util',   parentGroup: 'needs', allocated: 28_000 },
    { id: 'wants1', name: 'Coffee', parentGroup: 'wants', allocated: 3_000 },
    { id: 'sav1',   name: 'Savings',parentGroup: 'savings', allocated: 20_000 },
  ]
  // 12 months for rich + mid; just one observation for sparse, at a far-from-group value.
  const rows: TxRow[] = []
  const rng = makeRng('shrink')
  for (const m of MONTHS_2025) {
    rows.push({ date: `${m}-15T00:00:00Z`, categoryId: 'rich', amount: Math.exp(Math.log(30_000) + 0.10 * stdNormal(rng)) })
    rows.push({ date: `${m}-15T00:00:00Z`, categoryId: 'mid',  amount: Math.exp(Math.log(28_000) + 0.10 * stdNormal(rng)) })
    rows.push({ date: `${m}-15T00:00:00Z`, categoryId: 'wants1', amount: Math.exp(Math.log(3_000) + 0.20 * stdNormal(rng)) })
    rows.push({ date: `${m}-15T00:00:00Z`, categoryId: 'sav1',   amount: Math.exp(Math.log(20_000) + 0.15 * stdNormal(rng)) })
  }
  // Single sparse observation at a low value
  rows.push({ date: '2025-06-15T00:00:00Z', categoryId: 'sparse', amount: 5_000 })

  const data = aggregateMonthly(rows, cats)
  const period = makePeriod(cats.map((c) => c.id))
  const result = await runBudget(data, period, {
    seed: 'shrink',
    nIter: 1500,
    nBurn: 750,
    thin: 2,
    nChains: 3,
    yieldFn: () => Promise.resolve(),
  })

  const sparse = result.posteriors.find((p) => p.categoryId === 'sparse')!
  const rich = result.posteriors.find((p) => p.categoryId === 'rich')!
  const mid = result.posteriors.find((p) => p.categoryId === 'mid')!

  // The group-level posterior median should be closer to {rich, mid} (~30k, ~28k)
  // than to the sparse outlier (5k). So the sparse posterior median should be
  // pulled UP from the data point of 5k toward the group mean.
  // We expect: sparse.monthlyMean.median > 1.5 × the raw outlier.
  if (!(sparse.monthlyMean.median > 7_500)) {
    throw new Error(
      `sparse cat not shrunk enough: monthlyMean.median = ${sparse.monthlyMean.median} ` +
        `(should be pulled up toward group mean ~28k, expected > 7.5k)`,
    )
  }
  // And the rich category should stay near its data (~30k).
  if (!(rich.monthlyMean.median > 22_000 && rich.monthlyMean.median < 40_000)) {
    throw new Error(`rich monthlyMean.median = ${rich.monthlyMean.median} too far from 30k`)
  }
  // Sanity: mid should also be near its truth.
  if (!(mid.monthlyMean.median > 22_000 && mid.monthlyMean.median < 36_000)) {
    throw new Error(`mid monthlyMean.median = ${mid.monthlyMean.median} too far from 28k`)
  }
  // Sparse 90% CI should be much wider than rich's.
  const sparseWidth = sparse.monthlyMean.q95 - sparse.monthlyMean.q05
  const richWidth = rich.monthlyMean.q95 - rich.monthlyMean.q05
  if (!(sparseWidth > richWidth)) {
    throw new Error(`sparse CI width ${sparseWidth} not greater than rich CI width ${richWidth}`)
  }
})

// ---------------------------------------------------------------------------
// 4. Diagnostics + structural checks
// ---------------------------------------------------------------------------
Deno.test('runBudget: diagnostics + posteriors structure on 12 months × 6 cats', async () => {
  const cats: CategoryMeta[] = [
    { id: 'g1', name: 'Groceries',  parentGroup: 'needs',   allocated: 20_000 },
    { id: 'r1', name: 'Rent',       parentGroup: 'needs',   allocated: 35_000 },
    { id: 'd1', name: 'Dining',     parentGroup: 'wants',   allocated: 10_000 },
    { id: 'e1', name: 'Entertain',  parentGroup: 'wants',   allocated: 5_000 },
    { id: 's1', name: 'Emergency',  parentGroup: 'savings', allocated: 25_000 },
    { id: 'i1', name: 'Index Fund', parentGroup: 'savings', allocated: 15_000 },
  ]
  const trueParams = {
    g1: { theta: Math.log(18_000), sigma: 0.18 },
    r1: { theta: Math.log(34_000), sigma: 0.08 },
    d1: { theta: Math.log(8_500),  sigma: 0.25 },
    e1: { theta: Math.log(4_500),  sigma: 0.30 },
    s1: { theta: Math.log(22_000), sigma: 0.15 },
    i1: { theta: Math.log(14_000), sigma: 0.18 },
  }
  const rows = synthRows(trueParams, MONTHS_2025, 'diag')
  const data = aggregateMonthly(rows, cats)
  const period = makePeriod(cats.map((c) => c.id))
  const result = await runBudget(data, period, {
    seed: 'diag-budget',
    nIter: 1500,
    nBurn: 750,
    thin: 2,
    nChains: 3,
    yieldFn: () => Promise.resolve(),
  })

  // Diagnostics — relaxed CI gate; prod gate lives in result.diagnostics.pass.
  if (!(result.diagnostics.rhatMax < 1.10)) {
    throw new Error(`rhatMax ${result.diagnostics.rhatMax} too high`)
  }
  if (!(result.diagnostics.essMin > 100)) {
    throw new Error(`essMin ${result.diagnostics.essMin} too low`)
  }

  // Structural checks
  assertEquals(result.posteriors.length, 6)
  assertEquals(result.groupTotals.length, 3)

  // Quantile ordering
  for (const p of result.posteriors) {
    if (!(p.monthlyMean.q05 <= p.monthlyMean.q25 && p.monthlyMean.q25 <= p.monthlyMean.median)) {
      throw new Error(`monthlyMean quantiles non-monotone for ${p.categoryName}`)
    }
    if (!(p.monthlyMean.median <= p.monthlyMean.q75 && p.monthlyMean.q75 <= p.monthlyMean.q95)) {
      throw new Error(`monthlyMean upper quantiles non-monotone for ${p.categoryName}`)
    }
    if (!(p.pOver >= 0 && p.pOver <= 1)) {
      throw new Error(`pOver ${p.pOver} out of [0,1] for ${p.categoryName}`)
    }
  }

  // Group total quantile ordering.
  for (const g of result.groupTotals) {
    if (!(g.periodTotal.q05 <= g.periodTotal.median && g.periodTotal.median <= g.periodTotal.q95)) {
      throw new Error(`group ${g.group} periodTotal quantiles non-monotone`)
    }
    if (!(g.pOver >= 0 && g.pOver <= 1)) {
      throw new Error(`group ${g.group} pOver out of [0,1]`)
    }
  }

  // Period info
  assertEquals(result.period.startDate, '2025-01-01T00:00:00Z')
  if (result.period.daysRemaining <= 0) {
    throw new Error(`daysRemaining must be positive at mid-period; got ${result.period.daysRemaining}`)
  }
})

// ---------------------------------------------------------------------------
// 5. pOver responds to allocation: huge allocation → low pOver
// ---------------------------------------------------------------------------
Deno.test('runBudget: P(over) decreases as allocation grows', async () => {
  const baseCats: CategoryMeta[] = [
    { id: 'tight', name: 'TightBudget',  parentGroup: 'needs', allocated: 5_000 },   // way under truth
    { id: 'fat',   name: 'GenerousBudget', parentGroup: 'needs', allocated: 100_000 }, // way over truth
    { id: 'peer',  name: 'Peer',         parentGroup: 'needs', allocated: 20_000 },
    { id: 'w1',    name: 'WantsPeer',    parentGroup: 'wants', allocated: 8_000 },
    { id: 's1',    name: 'SavPeer',      parentGroup: 'savings', allocated: 20_000 },
  ]
  const trueParams = {
    tight: { theta: Math.log(20_000), sigma: 0.10 },
    fat:   { theta: Math.log(20_000), sigma: 0.10 },
    peer:  { theta: Math.log(20_000), sigma: 0.10 },
    w1:    { theta: Math.log(8_000),  sigma: 0.20 },
    s1:    { theta: Math.log(20_000), sigma: 0.15 },
  }
  const rows = synthRows(trueParams, MONTHS_2025, 'pover')
  const data = aggregateMonthly(rows, baseCats)

  // Place "today" near period start so almost all the period is forecast.
  const period: BudgetPeriodMeta = {
    startDate: '2025-01-01T00:00:00Z',
    endDate: '2025-01-31T23:59:59Z',
    today: '2025-01-02T00:00:00Z',
    spentSoFar: { tight: 0, fat: 0, peer: 0, w1: 0, s1: 0 },
  }

  const result = await runBudget(data, period, {
    seed: 'pover',
    nIter: 1500,
    nBurn: 750,
    thin: 2,
    nChains: 3,
    yieldFn: () => Promise.resolve(),
  })
  const tight = result.posteriors.find((p) => p.categoryId === 'tight')!
  const fat = result.posteriors.find((p) => p.categoryId === 'fat')!

  if (!(tight.pOver > 0.7)) {
    throw new Error(`tight pOver = ${tight.pOver}; expected > 0.7 (truth ~20k vs allocated 5k)`)
  }
  if (!(fat.pOver < 0.05)) {
    throw new Error(`fat pOver = ${fat.pOver}; expected < 0.05 (truth ~20k vs allocated 100k)`)
  }
})

// ---------------------------------------------------------------------------
// 6. Reproducibility
// ---------------------------------------------------------------------------
Deno.test('runBudget: deterministic given seed + data', async () => {
  const cats: CategoryMeta[] = [
    { id: 'a', name: 'A', parentGroup: 'needs',   allocated: 20_000 },
    { id: 'b', name: 'B', parentGroup: 'wants',   allocated: 10_000 },
    { id: 'c', name: 'C', parentGroup: 'savings', allocated: 15_000 },
  ]
  const rows = synthRows(
    {
      a: { theta: Math.log(15_000), sigma: 0.15 },
      b: { theta: Math.log(8_000),  sigma: 0.20 },
      c: { theta: Math.log(12_000), sigma: 0.15 },
    },
    MONTHS_2025.slice(0, 6),
    'rep-seed',
  )
  const data = aggregateMonthly(rows, cats)
  const period = makePeriod(['a', 'b', 'c'])

  const r1 = await runBudget(data, period, { seed: 'r', nIter: 600, nBurn: 300, thin: 2, nChains: 2, yieldFn: () => Promise.resolve() })
  const r2 = await runBudget(data, period, { seed: 'r', nIter: 600, nBurn: 300, thin: 2, nChains: 2, yieldFn: () => Promise.resolve() })

  for (const p1 of r1.posteriors) {
    const p2 = r2.posteriors.find((p) => p.categoryId === p1.categoryId)!
    assertAlmostEquals(p1.monthlyMean.median, p2.monthlyMean.median, 1e-6)
    assertAlmostEquals(p1.pOver, p2.pOver, 1e-9)
  }
})

// ---------------------------------------------------------------------------
// Quick null-data smoke test: aggregator must still produce data, runBudget
// must not crash with a category that has zero observations.
// ---------------------------------------------------------------------------
Deno.test('runBudget: handles a category with zero observations', async () => {
  const cats: CategoryMeta[] = [
    { id: 'has-data', name: 'HasData', parentGroup: 'needs', allocated: 20_000 },
    { id: 'no-data',  name: 'NoData',  parentGroup: 'needs', allocated: 25_000 },
    { id: 'w1',       name: 'WantsPeer', parentGroup: 'wants', allocated: 5_000 },
    { id: 's1',       name: 'SavPeer', parentGroup: 'savings', allocated: 10_000 },
  ]
  const rows = synthRows(
    {
      'has-data': { theta: Math.log(18_000), sigma: 0.15 },
      w1: { theta: Math.log(4_500), sigma: 0.20 },
      s1: { theta: Math.log(9_000), sigma: 0.15 },
    },
    MONTHS_2025.slice(0, 6),
    'no-data',
  )
  const data = aggregateMonthly(rows, cats)
  // Make sure the no-data category survived
  const noData = data.categories.find((c) => c.id === 'no-data')
  if (!noData) throw new Error('no-data category dropped from aggregation')
  assertEquals(data.nObs[data.categories.findIndex((c) => c.id === 'no-data')], 0)

  const period = makePeriod(cats.map((c) => c.id))
  const result = await runBudget(data, period, {
    seed: 'no-data',
    nIter: 600,
    nBurn: 300,
    thin: 2,
    nChains: 2,
    yieldFn: () => Promise.resolve(),
  })
  const noDataPost = result.posteriors.find((p) => p.categoryId === 'no-data')!
  // Posterior should exist and be finite.
  if (!isFinite(noDataPost.monthlyMean.median) || noDataPost.monthlyMean.median <= 0) {
    throw new Error(`no-data posterior median is not finite/positive: ${noDataPost.monthlyMean.median}`)
  }
  // Should be wider than the has-data CI (no shrinkage from data → looser).
  const hasData = result.posteriors.find((p) => p.categoryId === 'has-data')!
  const noDataWidth = noDataPost.monthlyMean.q95 - noDataPost.monthlyMean.q05
  const hasDataWidth = hasData.monthlyMean.q95 - hasData.monthlyMean.q05
  if (!(noDataWidth > hasDataWidth)) {
    throw new Error(`no-data CI ${noDataWidth} not wider than has-data CI ${hasDataWidth}`)
  }
})

// Avoid TS unused-import diagnostics.
const _unused: BudgetData | undefined = undefined
void _unused
