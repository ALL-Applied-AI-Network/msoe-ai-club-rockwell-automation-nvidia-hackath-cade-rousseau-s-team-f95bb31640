import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

// No cacheComponents / partialPrefetching: they need a Next.js server, and we ship a static export.
const base: NextConfig = {
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default function config(phase: string): NextConfig {
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    // `bun dev` forwards API calls to `uv run uvicorn api.main:app` on port 8000
    return {
      ...base,
      rewrites: async () => [{ source: "/api/:path*", destination: "http://localhost:8000/api/:path*" }],
    };
  }
  // `bun run build` writes static files to out/, which the FastAPI app serves at /
  return { ...base, output: "export" };
}
