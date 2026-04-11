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

   **Compose (persistent ADMIXTURE jobs):** from repo root, `docker compose up -d --build`. Set `INTERNAL_API_KEY` (and optional `ADMIXTURE_EXTRA_PATH`, binary mount) in a `.env` file; see `docker-compose.yml` comments.

3. **Use from frontend:** `https://YOUR_SERVER_IP:8000`  
   - Open port 8000 in the VPS firewall.  
   - For HTTPS, put Caddy or Nginx in front (e.g. Caddy: `caddy reverse-proxy --from yourdomain.com --to localhost:8000`).

### Private endpoint auth (required)

Paid conversion endpoints now require header **`X-Internal-Api-Key`**.

- Protected routes: `POST /raw-to-k36`, `POST /k36-to-g25`, `POST /raw-to-g25`, `POST /raw-to-g25/stream`, `POST /admixture/jobs`, `GET /admixture/jobs/{job_id}`, `POST /qpadm/jobs`, `GET /qpadm/jobs/{job_id}`
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

### ADMIXTURE jobs

The image does **not** bundle **ADMIXTURE**. Install the `admixture` binary on the host (or mount it) and set **`ADMIXTURE_EXTRA_PATH`** so the worker can find it (see `docker-compose.yml`).

- **`POST /admixture/jobs`** — **Multipart form** (same URL for both modes):
  - **Server PLINK (no big upload):** send `plink_prefix`, `k`, optional `threads`, optional `cross_validation` — **omit** the `bundle` file. Uses **`ADMIXTURE_HOST_PLINK_ROOT`** (bind-mount). Response includes `job_kind`: `host_disk`.
  - **Zip upload:** same form fields **plus** file `bundle` = zip with **`{plink_prefix}.bed`**, **`.bim`**, **`.fam`** at zip root. Response `job_kind`: `bundle`.
- **`GET /admixture/jobs/{job_id}`** — `job_kind`, `status`, `error`, `result` (`returncode`, `stdout`, `stderr`, `output_files`, `command`, `input_bed`, …).

| Variable | Default | Meaning |
|----------|---------|---------|
| `ADMIXTURE_ENABLED` | `true` | `false` disables `/admixture/*` and the background worker |
| `ADMIXTURE_MAX_BUNDLE_MB` | `6144` | Max zip size (large PLINK `.bed` sets need headroom) |
| `ADMIXTURE_TIMEOUT_SEC` | `86400` | Subprocess timeout (24h) |
| `ADMIXTURE_BIN` | `admixture` | Executable name or path |
| `ADMIXTURE_JOBS_ROOT` | `<repo>/admixture_jobs_data` | Job storage (use a volume on VPS) |
| `MAX_CONCURRENT_ADMIXTURE` | `1` | Parallel ADMIXTURE processes |
| `ADMIXTURE_OUTPUT_READ_MAX` | `2097152` | Max bytes of stdout/stderr and text outputs in `result` |
| `ADMIXTURE_HOST_PLINK_ROOT` | `/var/admixture/plink` (compose) | In-container path to read-only PLINK files; host dir must be mounted to match |

### qpAdm (ADMIXTOOLS 2) jobs

The Docker image includes **R** and the **ADMIXTOOLS 2** R package. Mount AADR reference data under **`/var/qpadm/ref`**.

#### One-time setup: precompute f2 blocks

After deploying and mounting the AADR panel, run `extract_f2.R` **once** to precompute blocked f2 statistics. This can take **hours** on a small VPS but makes `allsnps: false` qpAdm jobs complete in **seconds**.

**Memory:** full public AADR has **thousands** of populations × **~1.2M SNPs** — the allele-frequency step can need **tens of GB RAM** if chunking does not shrink it enough. On a typical VPS you should either:

- use **`--pops-file`** listing only populations your product allows (one label per line, matching `.ind` column 3), or  
- use aggressive chunking: **`--maxmem 1024 --cols_per_chunk 2`** (slower, less RAM), or  
- run extraction on a **64GB+** machine once, then copy the `/data/qpadm_f2` volume to production.

The script defaults to `fst=FALSE` and `afprod=FALSE` (f2-only), matching the API’s `f2_from_precomp(..., afprod=FALSE)` path.

```bash
# Example: subset pops (recommended on low-RAM VPS)
docker exec -it admix-api Rscript /app/scripts/extract_f2.R \
  /var/qpadm/ref/AADR/v62.0_1240k_public \
  /data/qpadm_f2 \
  --pops-file /var/qpadm/ref/allowed_pops.txt \
  --n_cores 4 --maxmem 2048 --cols_per_chunk 4

# Example: all pops from .ind (needs large RAM or small cols_per_chunk)
docker exec -it admix-api Rscript /app/scripts/extract_f2.R \
  /var/qpadm/ref/AADR/v62.0_1240k_public \
  /data/qpadm_f2 \
  --n_cores 4 --maxmem 1024 --cols_per_chunk 2
```

Re-run when the reference panel changes (new AADR version). **Jobs that request populations missing from the f2 cache will fail** until you re-extract with those pops included (or they fall back to the slow genotype path when `allsnps: true` only — for `allsnps: false` with f2 dir set, pops must exist in the cache).

#### API

- **`POST /qpadm/jobs`** — JSON body:
  - **`left_pops`**, **`right_pops`**: arrays of population labels (`.ind` column 3), one model per request; first **left** pop = target, rest = sources; **right** = outgroups.
  - **`genotypename`**, **`snpname`**, **`indivname`**: optional absolute paths inside an allowed prefix; if omitted, **`QPADM_DEFAULT_*`** env paths are used.
  - **`allsnps`**, **`inbreed`**, **`details`**: booleans.
- **`GET /qpadm/jobs/{job_id}`** — `status`, `error`, `result`.

When `result` is available (status `done`), it contains structured JSON:

```json
{
  "weights": [{"target": "X", "left": "A", "weight": 0.45, "se": 0.03, "z": 15.0}],
  "rankdrop": [{"f4rank": 0, "dof": 10, "chisq": 8.2, "p": 0.61}],
  "popdrop": [...],
  "elapsed_sec": 3.2
}
```

| Variable | Default (compose) | Meaning |
|----------|-------------------|---------|
| `QPADM_ENABLED` | `true` | `false` disables `/qpadm/*` and worker |
| `QPADM_JOBS_ROOT` | `/data/qpadm_jobs` | Job dirs + SQLite (volume) |
| `QPADM_F2_DIR` | `/data/qpadm_f2` | Precomputed f2 blocks from `extract_f2.R` |
| `QPADM_TIMEOUT_SEC` | `86400` | Subprocess timeout |
| `QPADM_ALLOWED_PATH_PREFIXES` | `/var/qpadm/ref` | Colon-separated roots; EIGENSTRAT paths must resolve under one |
| `QPADM_DEFAULT_GENO` / `_SNP` / `_IND` | AADR 1240k public paths | Used when request omits all three paths; also needed for `extract_f2.R` |
| `QPADM_RSCRIPT` | `Rscript` | Path to `Rscript` binary (usually default is fine) |

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
