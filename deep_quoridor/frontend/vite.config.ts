import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { viteStaticCopy } from "vite-plugin-static-copy";

// Copy onnxruntime-web's wasm/mjs runtime into the build so it is served
// SAME-ORIGIN (COEP require-corp, set by the Python server, blocks cross-origin
// non-CORP resources — a CDN load of ORT would be blocked). We point
// ort.env.wasm.wasmPaths at "/ort/" (see ai.worker.ts).
export default defineConfig({
  plugins: [
    svelte(),
    viteStaticCopy({
      targets: [
        { src: "node_modules/onnxruntime-web/dist/*.wasm", dest: "ort" },
        { src: "node_modules/onnxruntime-web/dist/*.mjs", dest: "ort" },
      ],
    }),
  ],
  // Vite serves cross-origin-isolation headers in dev so SharedArrayBuffer /
  // the wasm-CPU fallback work locally the same as behind the Python server.
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  worker: { format: "es" },
});
