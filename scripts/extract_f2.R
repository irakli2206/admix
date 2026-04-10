#!/usr/bin/env Rscript
#
# One-time f2 extraction for ADMIXTOOLS 2.
#
# Usage:
#   Rscript extract_f2.R <geno_prefix> <outdir> [--n_cores N] [--maxmem N] [--overwrite]
#
# <geno_prefix> is the EIGENSTRAT/PACKEDANCESTRYMAP prefix, e.g.
#   /var/qpadm/ref/AADR/v62.0_1240k_public
#   (expects .geno, .snp, .ind with that prefix)
#
# <outdir> is where the f2 blocks will be written.

library(admixtools)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript extract_f2.R <geno_prefix> <outdir> [--n_cores N] [--maxmem N] [--overwrite]")
}

pref   <- args[1]
outdir <- args[2]

n_cores   <- 1
maxmem    <- 8000
overwrite <- FALSE

i <- 3
while (i <= length(args)) {
  if (args[i] == "--n_cores" && i < length(args)) {
    n_cores <- as.integer(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--maxmem" && i < length(args)) {
    maxmem <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--overwrite") {
    overwrite <- TRUE
    i <- i + 1
  } else {
    stop(paste("Unknown argument:", args[i]))
  }
}

ind_file <- paste0(pref, ".ind")
if (!file.exists(ind_file)) {
  stop(paste("Cannot find .ind file:", ind_file))
}

ind <- read.table(ind_file, header = FALSE, stringsAsFactors = FALSE,
                  comment.char = "")
pops <- unique(ind[[3]])
cat(sprintf("Found %d unique populations in %s\n", length(pops), ind_file))
cat(sprintf("Output dir: %s\n", outdir))
cat(sprintf("n_cores=%d  maxmem=%.0f  overwrite=%s\n", n_cores, maxmem, overwrite))

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat("Starting f2 extraction...\n")
t0 <- proc.time()

extract_f2(
  pref,
  outdir,
  pops      = pops,
  maxmem    = maxmem,
  n_cores   = n_cores,
  overwrite = overwrite
)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done. f2 extraction took %.1f seconds (%.1f minutes).\n",
            elapsed, elapsed / 60))
