/** @type {import('next').NextConfig} */
const nextConfig = {
  // Force legacy webpack dev server instead of Turbopack
  experimental: {
    turbopack: false,
  },
  // Fix for react-pdf and canvas in Next.js
  webpack: (config: any) => {
    config.resolve.alias.canvas = false;
    config.resolve.alias.encoding = false;
    return config;
  },
};

export default nextConfig;
