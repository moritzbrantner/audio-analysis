# audio-analysis-recognition-server

Thin HTTP API adapter for `moritzbrantner-audio-analysis-recognition`.

Run:

```bash
cargo run -p audio-analysis-recognition-server -- --addr 127.0.0.1:3000
```

Endpoints:

- `GET /health`
- `GET /api/package`
- `GET /api/schema`
- `GET /api/operations`
- `POST /api/run`
- `POST /api/<operation-id>`
