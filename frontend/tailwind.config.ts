import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#14211b",
          soft: "#24352c",
          mute: "#5c6f64",
        },
        paper: {
          DEFAULT: "#f3f6f4",
          card: "#ffffff",
          line: "#d5ddd8",
        },
        accent: {
          DEFAULT: "#0f766e",
          soft: "#ccfbf1",
          deep: "#115e59",
        },
        warn: {
          DEFAULT: "#b45309",
          soft: "#ffedd5",
        },
      },
      fontFamily: {
        display: ["var(--font-display)", "Georgia", "serif"],
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 rgba(20, 33, 27, 0.06)",
      },
    },
  },
  plugins: [],
} satisfies Config;
