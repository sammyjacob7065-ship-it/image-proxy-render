# Image Proxy for Render

Thin proxy that calls free Hugging Face Flux Spaces with automatic fallback.

## Endpoints
- `GET /` → health check
- `POST /generate` → body: `{"prompt": "your text here"}`
