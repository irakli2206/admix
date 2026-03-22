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
