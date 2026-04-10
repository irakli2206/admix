#!/usr/bin/env Rscript
#
# Per-job qpAdm runner for ADMIXTOOLS 2.
#
# Usage:
#   Rscript run_qpadm.R <request.json>
#
# request.json fields:
#   left_pops     - array of population labels (first = target, rest = sources)
#   right_pops    - array of outgroup labels
#   f2_dir        - path to precomputed f2 blocks (from extract_f2.R)
#   geno_prefix   - (optional) EIGENSTRAT prefix; used as fallback if f2_dir missing
#   allsnps       - boolean
#   details       - boolean (if true, return f4 block details)
#
# Writes structured JSON to stdout. Diagnostics go to stderr.

suppressPackageStartupMessages(library(admixtools))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript run_qpadm.R <request.json>")
}

req <- fromJSON(args[1])

left  <- req$left_pops
right <- req$right_pops
if (is.null(left) || length(left) < 1)  stop("left_pops required")
if (is.null(right) || length(right) < 1) stop("right_pops required")

target  <- left[1]
sources <- left[-1]
if (length(sources) < 1) stop("Need at least 2 left_pops (target + 1 source)")

f2_dir      <- req$f2_dir
geno_prefix <- req$geno_prefix
allsnps     <- isTRUE(req$allsnps)
return_f4   <- isTRUE(req$details)

all_pops <- unique(c(left, right))

# Decide data source: precomputed f2 or genotype prefix fallback
if (!is.null(f2_dir) && dir.exists(f2_dir)) {
  message(sprintf("Using precomputed f2 from: %s", f2_dir))
  data_arg <- f2_dir
} else if (!is.null(geno_prefix)) {
  message(sprintf("No f2 dir; falling back to genotype prefix: %s", geno_prefix))
  data_arg <- geno_prefix
} else {
  stop("Neither f2_dir nor geno_prefix provided")
}

message(sprintf("target=%s  sources=%s  right=%s  allsnps=%s",
                target, paste(sources, collapse=","),
                paste(right, collapse=","), allsnps))

t0 <- proc.time()

res <- tryCatch({
  qpadm(
    data       = data_arg,
    left       = left,
    right      = right,
    target     = target,
    return_f4  = return_f4,
    verbose    = FALSE
  )
}, error = function(e) {
  result <- list(
    error = conditionMessage(e)
  )
  cat(toJSON(result, auto_unbox = TRUE, pretty = TRUE, na = "null"), "\n")
  quit(status = 1)
})

elapsed <- (proc.time() - t0)["elapsed"]
message(sprintf("qpadm() completed in %.2f seconds", elapsed))

# Build structured output
output <- list()

if (!is.null(res$weights)) {
  output$weights <- res$weights
}

if (!is.null(res$rankdrop)) {
  output$rankdrop <- res$rankdrop
}

if (!is.null(res$popdrop)) {
  output$popdrop <- res$popdrop
}

if (return_f4 && !is.null(res$f4)) {
  output$f4 <- res$f4
}

output$elapsed_sec <- round(as.numeric(elapsed), 3)

cat(toJSON(output, auto_unbox = TRUE, pretty = TRUE, na = "null",
           dataframe = "rows"), "\n")
