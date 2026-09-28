import type { NextConfig } from "next";

// FastAPI routes are unprefixed at :8000; the browser only ever talks to
// same-origin /api/* (no CORS). Override BACKEND_URL for the Playwright
// stack (e.g. a test backend on a different port).
const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/:path*`,
      },
    ];
  },

  async redirects() {
    return [
      { source: "/workbench", destination: "/", permanent: false },
      { source: "/gallery", destination: "/", permanent: false },
      { source: "/workbench/:projectId", destination: "/projects/:projectId", permanent: false },
      {
        source: "/workbench/:projectId/:interviewId",
        destination: "/projects/:projectId/interviews/:interviewId",
        permanent: false,
      },
      { source: "/gallery/personas/:projectId", destination: "/projects/:projectId/personas", permanent: false },
      {
        source: "/gallery/personas/:projectId/:personId",
        destination: "/projects/:projectId/personas/:personId",
        permanent: false,
      },
      { source: "/gallery/persons/:projectId", destination: "/projects/:projectId/people", permanent: false },
      {
        source: "/gallery/persons/:projectId/:personId",
        destination: "/projects/:projectId/people/:personId",
        permanent: false,
      },
      {
        source: "/gallery/worklist",
        has: [{ type: "query", key: "project", value: "(?<project>.+)" }],
        destination: "/projects/:project/review",
        permanent: false,
      },
      { source: "/gallery/worklist", destination: "/", permanent: false },
    ];
  },
};

export default nextConfig;
