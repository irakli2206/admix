#!/usr/bin/env Rscript
#
# One-time f2 extraction for ADMIXTOOLS 2.
#
# Usage:
#   Rscript extract_f2.R <geno_prefix> <outdir> [options]
#
# Options:
#   --n_cores N          parallel workers (default 1)
#   --maxmem MB          memory hint for chunking (default 2048; lower = smaller chunks)
#   --cols_per_chunk N   SNP chunks on disk (default 4; lower = less RAM, slower)
#   --pops-file PATH     one population label per line; omit to use ALL pops from .ind
#   --overwrite
#
# Full AADR (~4300 pops × 1.2M SNPs) needs either a large-RAM machine (64GB+)
# or a --pops-file subset. Typical VPS: use --pops-file with pops your app offers.

library(admixtools)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop(paste(
    "Usage: Rscript extract_f2.R <geno_prefix> <outdir> [options]\n",
    "  --n_cores N  --maxmem MB  --cols_per_chunk N  --pops-file PATH  --overwrite"
  ))
}

pref   <- args[1]
outdir <- args[2]

n_cores         <- 1
maxmem          <- 2048
cols_per_chunk  <- 4
overwrite       <- FALSE
pops_file       <- NULL

i <- 3
while (i <= length(args)) {
  if (args[i] == "--n_cores" && i < length(args)) {
    n_cores <- as.integer(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--maxmem" && i < length(args)) {
    maxmem <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--cols_per_chunk" && i < length(args)) {
    cols_per_chunk <- as.integer(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--pops-file" && i < length(args)) {
    pops_file <- args[i + 1]
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
if (!is.null(pops_file)) {
  if (!file.exists(pops_file)) stop(paste("pops-file not found:", pops_file))
  pops <- readLines(pops_file, warn = FALSE)
  pops <- unique(trimws(pops[nchar(trimws(pops)) > 0]))
  avail <- unique(ind[[3]])
  miss <- setdiff(pops, avail)
  if (length(miss) > 0) {
    stop(paste(
      length(miss), "labels in pops-file are not in .ind col3, e.g.:",
      paste(head(miss, 5), collapse = ", ")
    ))
  }
  cat(sprintf("Using %d populations from %s\n", length(pops), pops_file))
} else {
  pops <- unique(ind[[3]])
  cat(sprintf("Found %d unique populations in %s\n", length(pops), ind_file))
  if (length(pops) > 800) {
    message(
      "NOTE: Very many populations — RAM use is huge unless chunking works.\n",
      "      If this fails with 'cannot allocate vector', use --pops-file with a subset,\n",
      "      or run on a host with 64GB+ RAM, or try --maxmem 1024 --cols_per_chunk 2."
    )
  }
}

cat(sprintf("Output dir: %s\n", outdir))
cat(sprintf(
  "n_cores=%d maxmem=%.0f MB cols_per_chunk=%d fst=FALSE afprod=FALSE overwrite=%s\n",
  n_cores, maxmem, cols_per_chunk, overwrite
))

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat("Starting f2 extraction...\n")
t0 <- proc.time()

extract_f2(
  pref,
  outdir,
  pops             = pops,
  maxmem           = maxmem,
  cols_per_chunk   = cols_per_chunk,
  fst              = FALSE,
  afprod           = FALSE,
  n_cores          = n_cores,
  overwrite        = overwrite,
  verbose          = TRUE
)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done. f2 extraction took %.1f seconds (%.1f minutes).\n",
            elapsed, elapsed / 60))
