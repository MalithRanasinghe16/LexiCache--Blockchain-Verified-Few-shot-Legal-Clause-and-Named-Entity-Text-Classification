/** @type {import('next').NextConfig} */
const nextConfig = {
  // Force legacy webpack dev server instead of Turbopack
  experimental: {
    turbopack: false,
  },
};

export default nextConfig;