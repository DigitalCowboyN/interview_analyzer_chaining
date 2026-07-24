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
};

export default nextConfig;
