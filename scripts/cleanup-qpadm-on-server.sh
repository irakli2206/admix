#!/usr/bin/env bash
# One-time cleanup after migrating away from qpAdm (run on the VPS as root or docker group).
# Stops the stack, removes the old named volume, optional host dirs. Review before running.

set -euo pipefail

REPO="${ADMIX_DIR:-$HOME/admix}"
cd "$REPO" || {
  echo "Repo not found: $REPO — set ADMIX_DIR"
  exit 1
}

echo "Stopping compose in $REPO ..."
docker compose down 2>/dev/null || true

echo "Docker volumes matching *qpadm* (remove manually if listed):"
docker volume ls | grep -i qpadm || echo "(none)"

# Typical compose project name + volume from old stack:
for v in admix_qpadm_jobs root_qpadm_jobs qpadm_jobs; do
  if docker volume inspect "$v" &>/dev/null; then
    echo "Removing volume: $v"
    docker volume rm "$v"
  fi
done

echo "Optional: remove host paths if you no longer need them (uncomment to apply):"
echo "# rm -rf /var/qpadm/ref"
echo "# rm -rf /opt/admixtools/src/bin   # only if nothing else uses ADMIXTOOLS"
echo "# rm -rf $REPO/qpadm_jobs"

echo "Done. Redeploy with current docker-compose (ADMIXTURE) and place admixture binary under /opt/admixture on the host."
