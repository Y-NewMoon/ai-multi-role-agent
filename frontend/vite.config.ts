import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 前端开发服务器把 /api 代理到本地 FastAPI，避免跨域；
// SSE 走同一代理，http-proxy 默认透传流式响应（后端已带 X-Accel-Buffering: no）。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // 头像静态资源（M5）：后端 /static/avatars/*，开发期经同一代理访问
      '/static': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
