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
