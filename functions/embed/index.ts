// Supabase Edge Function: embed
// ============================================================================
// Wraps the built-in `Supabase.ai.Session('gte-small')` model to expose a
// minimal HTTP endpoint for batched 384-dim embeddings. Used by:
//   - /api/cron/embed-pending (backfill)
//   - /api/categorize/suggest (L3 path; lands in PR 5)
//
// Request  POST /functions/v1/embed  Authorization: Bearer <jwt>
//   { "texts": ["foodpanda order", "chai", ...] }
//
// Response 200
//   { "embeddings": [[...384 floats...], [...], ...] }
//
// JWT verification is ON by default. Service-role JWT works for
// server-to-server calls (cron + backfill); user JWT works for interactive
// paths (suggest route).
// ============================================================================

// @ts-expect-error — Deno + Supabase globals provided by the Edge Function runtime
const session = new Supabase.ai.Session('gte-small')

const MAX_BATCH = 64
const MAX_TEXT_LEN = 2000

interface EmbedRequest {
  texts: string[]
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

// @ts-expect-error — Deno global
Deno.serve(async (req: Request) => {
  if (req.method !== 'POST') {
    return json(405, { error: 'Method not allowed' })
  }

  let body: EmbedRequest
  try {
    body = (await req.json()) as EmbedRequest
  } catch {
    return json(400, { error: 'Invalid JSON body' })
  }

  if (!body.texts || !Array.isArray(body.texts)) {
    return json(400, { error: 'Missing or invalid `texts` array' })
  }
  if (body.texts.length === 0) {
    return json(200, { embeddings: [] })
  }
  if (body.texts.length > MAX_BATCH) {
    return json(400, { error: `Batch too large: max ${MAX_BATCH} texts per call` })
  }

  const trimmed = body.texts.map((t) => {
    if (typeof t !== 'string') return ' '
    const s = t.trim().slice(0, MAX_TEXT_LEN)
    return s.length > 0 ? s : ' '
  })

  const embeddings: number[][] = []
  for (const text of trimmed) {
    const out = (await session.run(text, {
      mean_pool: true,
      normalize: true,
    })) as number[]
    embeddings.push(out)
  }

  return json(200, { embeddings })
})
