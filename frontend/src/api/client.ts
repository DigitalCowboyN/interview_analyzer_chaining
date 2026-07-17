import { getCurrentUserId } from "@/identity/IdentityProvider";
import type { paths } from "./schema.d.ts";

/**
 * Typed API client. Every component/hook goes through here — never `fetch`
 * directly (loose-coupling rule: frontend/src/api → frontend/src/hooks →
 * components → routes, one direction only).
 *
 * Requests go to same-origin `/api/*`, which next.config.ts rewrites to the
 * FastAPI backend (no CORS). `X-User-ID` is attached to every request
 * (including GETs — harmless, consistent) from the identity store.
 */

const API_BASE = "/api";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(
      typeof detail === "string"
        ? detail
        : `Request failed with status ${status}`,
    );
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

type PathsWithMethod<Method extends string> = {
  [Path in keyof paths]: Method extends keyof paths[Path] ? Path : never;
}[keyof paths];

type SuccessResponse<T> = T extends { responses: infer R }
  ? R extends Record<number, unknown>
    ?
        | (200 extends keyof R ? R[200] : never)
        | (201 extends keyof R ? R[201] : never)
        | (202 extends keyof R ? R[202] : never)
    : never
  : never;

type JsonOf<T> = T extends { content: { "application/json": infer J } }
  ? J
  : never;

type ResponseBody<Path extends keyof paths, Method extends string> =
  Method extends keyof paths[Path]
    ? JsonOf<SuccessResponse<paths[Path][Method]>>
    : never;

interface RequestOptions {
  /** Path params to substitute into the URL template, e.g. { project_id: "p1" }. */
  params?: Record<string, string | number>;
  /** Query string params. */
  query?: Record<string, string | number | boolean | undefined>;
  /** JSON request body (POST/DELETE with body). */
  body?: unknown;
  signal?: AbortSignal;
}

function buildUrl(
  path: string,
  params?: RequestOptions["params"],
  query?: RequestOptions["query"],
): string {
  let url = path;
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      url = url.replace(`{${key}}`, encodeURIComponent(String(value)));
    }
  }
  if (query) {
    const search = new URLSearchParams();
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined) search.set(key, String(value));
    }
    const qs = search.toString();
    if (qs) url += `?${qs}`;
  }
  return `${API_BASE}${url}`;
}

async function request<T>(
  method: string,
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const url = buildUrl(path, options.params, options.query);
  const headers: Record<string, string> = {
    "X-User-ID": getCurrentUserId(),
  };
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(url, {
    method,
    headers,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
    signal: options.signal,
  });

  if (!response.ok) {
    let detail: unknown;
    try {
      const data = await response.json();
      detail = data?.detail ?? data;
    } catch {
      detail = response.statusText;
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

/** Typed GET keyed by an OpenAPI path from schema.d.ts. */
export function apiGet<Path extends PathsWithMethod<"get">>(
  path: Path,
  options?: RequestOptions,
): Promise<ResponseBody<Path, "get">> {
  return request(
    "GET",
    path as string,
    options,
  ) as Promise<ResponseBody<Path, "get">>;
}

/** Typed POST keyed by an OpenAPI path from schema.d.ts. */
export function apiPost<Path extends PathsWithMethod<"post">>(
  path: Path,
  options?: RequestOptions,
): Promise<ResponseBody<Path, "post">> {
  return request(
    "POST",
    path as string,
    options,
  ) as Promise<ResponseBody<Path, "post">>;
}

/** Typed DELETE keyed by an OpenAPI path from schema.d.ts. */
export function apiDelete<Path extends PathsWithMethod<"delete">>(
  path: Path,
  options?: RequestOptions,
): Promise<ResponseBody<Path, "delete">> {
  return request(
    "DELETE",
    path as string,
    options,
  ) as Promise<ResponseBody<Path, "delete">>;
}

/**
 * Low-level escape hatch for callers that need full control (e.g. the
 * intent-mutation wrapper in Task 5 that needs raw status codes to
 * distinguish 202/409/network failure). Prefer apiGet/apiPost/apiDelete.
 */
export async function apiFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("X-User-ID", getCurrentUserId());
  return fetch(`${API_BASE}${path}`, { ...init, headers });
}
