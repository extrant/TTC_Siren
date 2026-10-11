import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// The build goes to dist/, which the server hosts (Server:StaticDirectory, or wwwroot next to the program after the publish). File names carry a content hash, so the server
// can cache them for a year (immutable); only index.html is revalidated.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    assetsInlineLimit: 0, // nothing is inlined as a data: URL; the strict policy allows img data: but there is no need
    cssCodeSplit: true,
    chunkSizeWarningLimit: 1000, // the 3D table (three.js) is one lazy chunk of about 900 kB; the first page does not load it
    rolldownOptions: { output: { manualChunks: undefined } },
  },
  server: { proxy: { '/v1': 'http://127.0.0.1:5000', '/health': 'http://127.0.0.1:5000' } },
  test: { environment: 'jsdom', include: ['src/**/*.test.ts', 'src/**/*.test.tsx'], globals: false },
})
