"""Unit tests for qpAdm API helpers (no qpAdm binary)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from qpadm.eigenstrat_subset import subset_eigenstrat_by_pops
from qpadm.paths import resolve_under_allowed
from qpadm.workdir import validate_pop_token, write_job_workdir


class TestValidatePopToken(unittest.TestCase):
    def test_accepts_labels(self):
        self.assertEqual(validate_pop_token("Mbuti.DG"), "Mbuti.DG")
        self.assertEqual(validate_pop_token("  Foo.Bar  "), "Foo.Bar")

    def test_rejects_bad(self):
        with self.assertRaises(ValueError):
            validate_pop_token("")
        with self.assertRaises(ValueError):
            validate_pop_token("../evil")
        with self.assertRaises(ValueError):
            validate_pop_token("no spaces")


class TestResolveUnderAllowed(unittest.TestCase):
    def test_allows_under_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = os.path.join(tmp, "a.geno")
            open(g, "w").close()
            old = os.environ.get("QPADM_ALLOWED_PATH_PREFIXES")
            os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = tmp
            try:
                r = resolve_under_allowed(g)
                self.assertTrue(r.endswith("a.geno"))
            finally:
                if old is None:
                    os.environ.pop("QPADM_ALLOWED_PATH_PREFIXES", None)
                else:
                    os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = old


class TestWriteJobWorkdir(unittest.TestCase):
    def test_writes_par_and_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("x.geno", "x.snp", "x.ind"):
                open(os.path.join(tmp, name), "w").close()
            old = os.environ.get("QPADM_ALLOWED_PATH_PREFIXES")
            os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = tmp
            try:
                work = os.path.join(tmp, "job_work")
                snap = write_job_workdir(
                    work,
                    {
                        "left_pops": ["Target.Pop", "Source.Pop"],
                        "right_pops": ["Mbuti.DG"],
                        "ind_mode": "full",
                        "genotypename": os.path.join(tmp, "x.geno"),
                        "snpname": os.path.join(tmp, "x.snp"),
                        "indivname": os.path.join(tmp, "x.ind"),
                        "allsnps": False,
                        "inbreed": False,
                        "details": True,
                    },
                )
                self.assertIn("genotypename", snap)
                with open(os.path.join(work, "left_pops.txt"), encoding="utf-8") as f:
                    self.assertIn("Target.Pop", f.read())
                with open(os.path.join(work, "qpAdm.par"), encoding="utf-8") as f:
                    body = f.read()
                    self.assertIn("popleft: left_pops.txt", body)
                    self.assertIn("details: YES", body)
                with open(os.path.join(work, "request.json"), encoding="utf-8") as f:
                    json.load(f)
            finally:
                if old is None:
                    os.environ.pop("QPADM_ALLOWED_PATH_PREFIXES", None)
                else:
                    os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = old

    def test_custom_ind_mode_writes_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("x.geno", "x.snp", "x.ind"):
                open(os.path.join(tmp, name), "w").close()
            old = os.environ.get("QPADM_ALLOWED_PATH_PREFIXES")
            os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = tmp
            try:
                # 3 individuals, 2 SNPs; only pops A and C in model -> indices 0 and 2
                ind_path = os.path.join(tmp, "x.ind")
                with open(ind_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("i1 U A\ni2 U B\ni3 U C\n")
                snp_path = os.path.join(tmp, "x.snp")
                with open(snp_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("s1 1 0.0 0 A 1\ns2 1 0.0 0 A 1\n")
                geno_path = os.path.join(tmp, "x.geno")
                with open(geno_path, "wb") as f:
                    f.write(b"019\n928\n")
                work = os.path.join(tmp, "job_work")
                snap = write_job_workdir(
                    work,
                    {
                        "left_pops": ["A"],
                        "right_pops": ["C"],
                        "genotypename": geno_path,
                        "snpname": snp_path,
                        "indivname": ind_path,
                        "ind_mode": "custom",
                        "allsnps": False,
                        "inbreed": False,
                        "details": False,
                    },
                )
                self.assertEqual(snap["ind_mode"], "custom")
                self.assertIn("subset_source", snap)
                with open(os.path.join(work, "qpAdm.par"), encoding="utf-8") as f:
                    par = f.read()
                self.assertIn("subset.geno", par)
                with open(os.path.join(work, "subset.ind"), encoding="utf-8") as f:
                    sub_ind = f.read().strip().splitlines()
                self.assertEqual(len(sub_ind), 2)
                with open(os.path.join(work, "subset.geno"), "rb") as f:
                    rows = f.read().splitlines()
                self.assertEqual(rows, [b"09", b"98"])
            finally:
                if old is None:
                    os.environ.pop("QPADM_ALLOWED_PATH_PREFIXES", None)
                else:
                    os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = old


class TestEigenstratSubset(unittest.TestCase):
    def test_subset_matches_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            ind = os.path.join(tmp, "a.ind")
            with open(ind, "w", encoding="utf-8", newline="\n") as f:
                f.write("a U P\nb U Q\nc U P\n")
            snp = os.path.join(tmp, "a.snp")
            with open(snp, "w", encoding="utf-8", newline="\n") as f:
                f.write("s1 1 0 0 A 1\n")
            geno = os.path.join(tmp, "a.geno")
            with open(geno, "wb") as f:
                f.write(b"012\n")
            out = os.path.join(tmp, "out")
            g2, s2, i2 = subset_eigenstrat_by_pops(geno, snp, ind, {"P"}, out)
            self.assertTrue(os.path.isfile(g2))
            self.assertTrue(os.path.isfile(s2))
            self.assertTrue(os.path.isfile(i2))
            with open(g2, "rb") as f:
                self.assertEqual(f.read().strip(), b"02")

    def test_no_match_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ind = os.path.join(tmp, "a.ind")
            with open(ind, "w", encoding="utf-8", newline="\n") as f:
                f.write("a U P\n")
            snp = os.path.join(tmp, "a.snp")
            open(snp, "w").close()
            geno = os.path.join(tmp, "a.geno")
            with open(geno, "wb") as f:
                f.write(b"0\n")
            with self.assertRaises(ValueError):
                subset_eigenstrat_by_pops(geno, snp, ind, {"Z"}, tmp)

    def test_transposed_layout(self):
        """One geno row per individual; output is SNP-major for qpAdm."""
        with tempfile.TemporaryDirectory() as tmp:
            ind = os.path.join(tmp, "a.ind")
            with open(ind, "w", encoding="utf-8", newline="\n") as f:
                f.write("a U P\nb U Q\nc U P\n")
            snp = os.path.join(tmp, "a.snp")
            with open(snp, "w", encoding="utf-8", newline="\n") as f:
                f.write("s1 1 0 0 A 1\ns2 1 0 0 A 1\n")
            geno = os.path.join(tmp, "a.geno")
            with open(geno, "wb") as f:
                f.write(b"01\n10\n02\n")
            out = os.path.join(tmp, "out")
            g2, _, _ = subset_eigenstrat_by_pops(geno, snp, ind, {"P"}, out)
            with open(g2, "rb") as f:
                rows = f.read().splitlines()
            self.assertEqual(rows, [b"00", b"12"])


if __name__ == "__main__":
    unittest.main()
