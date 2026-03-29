# Deploy backend (cheapest, no RAM limit)

**Option: VPS ~$5/month, 2–6 GB RAM**

1. **Rent a VPS** (pick one):
   - [Contabo](https://contabo.com) — Cloud VPS S (e.g. 4 GB RAM) ~€5/month
   - [Hetzner](https://www.hetzner.com/cloud) — CX22 (2 GB RAM) ~€4/month

2. **On the server** (after SSH):

   ```bash
   # Install Docker (Ubuntu/Debian)
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   # log out and back in, then:

   # Clone your repo (or upload files)
   git clone <your-repo-url> admix && cd admix

   # Ensure data/ has K36.alleles and K36.36.F
   ls data/

   # Build and run
   docker build -t admix-api .
   docker run -d --restart unless-stopped -p 8000:8000 --name admix admix-api
   # Low RAM? Add: -e MAX_CONCURRENT_CONVERSIONS=1
   ```

3. **Use from frontend:** `https://YOUR_SERVER_IP:8000`  
   - Open port 8000 in the VPS firewall.  
   - For HTTPS, put Caddy or Nginx in front (e.g. Caddy: `caddy reverse-proxy --from yourdomain.com --to localhost:8000`).

### Private endpoint auth (required)

Paid conversion endpoints now require header **`X-Internal-Api-Key`**.

- Protected routes: `POST /raw-to-k36`, `POST /k36-to-g25`, `POST /raw-to-g25`, `POST /raw-to-g25/stream`
- Set backend env on run:

```bash
docker run -d --restart unless-stopped -p 8000:8000 \
  -e INTERNAL_API_KEY="<long-random-secret>" \
  --name admix admix-api
```

Call these endpoints from your frontend server/proxy (not directly from browser), and inject:

```http
X-Internal-Api-Key: <same-secret>
```

If missing/wrong, backend returns `401`. If `INTERNAL_API_KEY` is not configured, backend returns `503`.

### Progress (SSE) for a UI progress bar

`POST /raw-to-g25/stream` returns **`text/event-stream`**. Each line is `data: <JSON>\n\n` with fields like:

- **`progress`** — 0–100 (rough stages: read → genotypes → optimizer → G25)
- **`stage`** — short label (`read`, `genotypes`, `optimizer`, `g25_regression`, …)
- **`done`** — `true` on the final event (then **`result`** has the same payload as `POST /raw-to-g25`, or **`error`** on failure)

`EventSource` only supports GET, so use **`fetch()` + read the body stream** and parse `data:` lines. Example:

```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);
form.append("vendor", "23andme");
form.append("sample_name", "me");

const res = await fetch("/raw-to-g25/stream", { method: "POST", body: form });
const reader = res.body.getReader();
const dec = new TextDecoder();
let buf = "";
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += dec.decode(value, { stream: true });
  let i;
  while ((i = buf.indexOf("\n\n")) >= 0) {
    const block = buf.slice(0, i);
    buf = buf.slice(i + 2);
    const line = block.split("\n").find((l) => l.startsWith("data: "));
    if (!line) continue;
    const json = JSON.parse(line.slice(6).trim());
    if (json.progress != null) setProgressBar(json.progress);
    if (json.done && json.result) setFinalResult(json.result);
    if (json.done && json.error) showError(json.error);
  }
}
```

Swagger often won’t preview the stream; test with the snippet above or `curl -N`.

Done. One small server, no RAM limit, ~$5/month.

---

## Local Docker (faster iteration)

**Option A — one command, reload on save (best for editing code)**  
From the project folder:

```bash
docker compose -f docker-compose.dev.yml up --build
```

- Mounts your repo into the container and runs `uvicorn --reload`: **change `.py` → save → server restarts**, no rebuild.
- Use `--build` when you change the **Dockerfile** or need to refresh the image; otherwise `docker compose -f docker-compose.dev.yml up` is enough.

**Option B — classic two-step**  
Rebuilds are **cached**: only layers after a changed `COPY` rerun, so repeat builds are usually quick.

```bash
docker build -t admix-api .
docker run --rm -p 8000:8000 admix-api
```

**Option C — one line (PowerShell)**  
`docker build -t admix-api .; docker run --rm -p 8000:8000 admix-api`
