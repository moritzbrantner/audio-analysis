# audio-analysis-speakers-server

Thin HTTP API adapter for `moritzbrantner-audio-analysis-speakers`.

Run:

```bash
cargo run -p audio-analysis-speakers-server -- --addr 127.0.0.1:3000
```

Endpoints:

- `GET /health`
- `GET /api/package`
- `GET /api/schema`
- `GET /api/operations`
- `POST /api/run`
- `POST /api/<operation-id>`
