import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiPort = process.env.LUNARMIND_API_PORT || "8000";
const webPort = Number(process.env.LUNARMIND_WEB_PORT || 5173);

export default defineConfig({
  plugins: [react()],
  server: {
    port: webPort,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${apiPort}`,
        ws: true,
      },
    },
  },
});
