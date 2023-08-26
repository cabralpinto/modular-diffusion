import mdx from "@astrojs/mdx";
import tailwind from "@astrojs/tailwind";
import { defineConfig } from "astro/config";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import remarkLayout from "./src/plugins/remark-layout.mjs";

// https://astro.build/config
export default defineConfig({
  integrations: [tailwind(), mdx()],
  markdown: {
    remarkPlugins: [remarkMath, remarkLayout],
    rehypePlugins: [rehypeKatex],
  },
  site: "https://cabralpinto.github.io",
  base: "/modular-diffusion/docs",
  redirects: {
    "/": "/modular-diffusion/docs/guides/getting-started",
  },
});
