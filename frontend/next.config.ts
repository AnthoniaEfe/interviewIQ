import type { NextConfig } from "next";

// Public backend URL (RunPod HTTPS proxy in production). Used only for optional
// same-origin rewrites; the browser still calls NEXT_PUBLIC_API_URL directly.
const apiUrl = (
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const nextConfig: NextConfig = {
  // Frontend must stay free of PyTorch / embedding models — GPU work lives on RunPod.
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
