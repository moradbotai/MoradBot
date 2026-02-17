import type { NextConfig } from "next";

const nextConfig: NextConfig = {
	reactStrictMode: true,
	transpilePackages: ["@moradbot/shared"],
	outputFileTracingRoot: process.cwd(),
};

export default nextConfig;
