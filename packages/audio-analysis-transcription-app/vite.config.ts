import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const uiSourceRoot = fileURLToPath(new URL("../video-analysis-ui/src", import.meta.url));

export default defineConfig({
  optimizeDeps: {
    exclude: ["@moritzbrantner/audio-analysis-transcription-wasm"],
  },
  plugins: [react()],
  resolve: {
    alias: [
      { find: /^@video-analysis\/ui$/, replacement: `${uiSourceRoot}/index.ts` },
      { find: /^@video-analysis\/ui\/tailwind-content$/, replacement: `${uiSourceRoot}/tailwind-content.ts` },
      { find: /^@video-analysis\/ui\/([^/]+)$/, replacement: `${uiSourceRoot}/$1/index.tsx` },
    ],
  },
});
