"""CLI: python admix_cli.py -f <raw_file> -v 23andme -m K36"""

import sys
import argparse
import numpy as np

import admix_models
from admix_fraction import admix_fraction


def main():
    p = argparse.ArgumentParser(description="K36 admixture from raw DNA.")
    p.add_argument("-f", "--file", required=True, help="Raw genome file")
    p.add_argument("-v", "--vendor", default="23andme", help="Format: 23andme, ancestry, ftdna, ftdna2, wegene, myheritage")
    p.add_argument("-m", "--models", nargs="+", default=["K36"], help="Model(s), default K36")
    p.add_argument("-o", "--output", help="Write results to file")
    p.add_argument("-t", "--tolerance", default="1e-3", help="Optimization tolerance")
    p.add_argument("--sort", action="store_true", help="Sort by proportion descending")
    p.add_argument("--ignore-zeros", action="store_true", help="Hide zero proportions")
    args = p.parse_args()

    for m in args.models:
        if m not in admix_models.models():
            print("Unknown model:", m)
            sys.exit(1)

    tol = float(args.tolerance)
    out_file = open(args.output, "w") if args.output else None

    try:
        for model in args.models:
            frac = np.array(admix_fraction(model, args.vendor, args.file, tol))
            pops = np.array(admix_models.populations(model))
            if args.sort:
                idx = np.argsort(frac)[::-1]
                frac, pops = frac[idx], pops[idx]
            lines = [f"{pops[i][0]}: {100*frac[i]:.2f}%" for i in range(len(frac))
                     if not (args.ignore_zeros and frac[i] < 1e-4)]
            text = model + "\n" + "\n".join(lines) + "\n\n"
            print(text)
            if out_file:
                out_file.write(text)
        if out_file:
            print("Written to", args.output)
    finally:
        if out_file:
            out_file.close()


if __name__ == "__main__":
    main()
