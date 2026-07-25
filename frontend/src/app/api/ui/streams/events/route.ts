/**
 * SSE streaming proxy for the live-feed endpoint (M5.1).
 *
 * WHY THIS EXISTS (not a Next.js `rewrites()` entry like every other
 * `/api/*` call): Next's rewrite proxy BUFFERS the upstream response — it
 * waits for the body to complete before forwarding anything to the client.
 * An SSE stream never completes, so through a rewrite the browser never
 * receives the 200 headers, `EventSource.onopen` never fires, and the live
 * indicator stays "off". Verified against both `next dev` and `next start`:
 * a rewrite yields zero bytes; this route handler streams the body through
 * incrementally. The design spec deferred "a BFF route handler" as
 * "nothing needs before M5.1" — streaming SSE is exactly that need.
 *
 * A route handler at this path takes precedence over the `/api/:path*`
 * rewrite (Next resolves filesystem/app routes before afterFiles rewrites),
 * so the rewrite still carries every other `/api/*` call unchanged; only
 * this one streaming endpoint is special-cased. Loose coupling holds: the
 * browser still talks only to same-origin `/api/ui/streams/events`, the
 * backend origin stays unexposed, no CORS.
 */

// Never statically optimize or cache — this is an infinite live stream.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function GET(request: Request): Promise<Response> {
  const incoming = new URL(request.url);
  const target = new URL(`${BACKEND_URL}/ui/streams/events`);
  // Forward only the contract's query params; the backend validates them
  // (≥1 required → 422), which we surface verbatim.
  for (const key of ["interview_id", "project_id"]) {
    const value = incoming.searchParams.get(key);
    if (value !== null) target.searchParams.set(key, value);
  }

  const upstream = await fetch(target, {
    headers: { Accept: "text/event-stream" },
    // Abort the upstream fetch when the browser disconnects, so the backend
    // subscription closes and the watcher can reach its zero-subscriber stop.
    signal: request.signal,
    // Opt out of Next's data cache (streaming itself comes from passing
    // upstream.body through untouched, below).
    cache: "no-store",
  });

  if (!upstream.ok || upstream.body === null) {
    // Pass the backend error through, keeping its content-type so a 422's
    // JSON `detail` is legible to anyone inspecting the response.
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
    });
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
