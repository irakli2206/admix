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
            old_prefix = os.environ.get("QPADM_ALLOWED_PATH_PREFIXES")
            old_root = os.environ.get("QPADM_JOBS_ROOT")
            os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = tmp
            os.environ["QPADM_JOBS_ROOT"] = os.path.join(tmp, "jobs_root")
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
                geno_in_par = snap["genotypename"]
                with open(geno_in_par, "rb") as f:
                    rows = f.read().splitlines()
                self.assertEqual(rows, [b"09", b"98"])
            finally:
                if old_prefix is None:
                    os.environ.pop("QPADM_ALLOWED_PATH_PREFIXES", None)
                else:
                    os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = old_prefix
                if old_root is None:
                    os.environ.pop("QPADM_JOBS_ROOT", None)
                else:
                    os.environ["QPADM_JOBS_ROOT"] = old_root


    def test_custom_ind_mode_cache_hit(self):
        """Second call with same pops should use cached subset (no re-read)."""
        with tempfile.TemporaryDirectory() as tmp:
            old_prefix = os.environ.get("QPADM_ALLOWED_PATH_PREFIXES")
            old_root = os.environ.get("QPADM_JOBS_ROOT")
            os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = tmp
            os.environ["QPADM_JOBS_ROOT"] = os.path.join(tmp, "jobs_root")
            try:
                ind_path = os.path.join(tmp, "x.ind")
                with open(ind_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("i1 U A\ni2 U B\ni3 U C\n")
                snp_path = os.path.join(tmp, "x.snp")
                with open(snp_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("s1 1 0.0 0 A 1\n")
                geno_path = os.path.join(tmp, "x.geno")
                with open(geno_path, "wb") as f:
                    f.write(b"019\n")
                body = {
                    "left_pops": ["A"],
                    "right_pops": ["C"],
                    "genotypename": geno_path,
                    "snpname": snp_path,
                    "indivname": ind_path,
                    "ind_mode": "custom",
                    "allsnps": False,
                    "inbreed": False,
                    "details": False,
                }
                work1 = os.path.join(tmp, "job1")
                snap1 = write_job_workdir(work1, body)
                work2 = os.path.join(tmp, "job2")
                snap2 = write_job_workdir(work2, body)
                self.assertEqual(snap1["genotypename"], snap2["genotypename"])
            finally:
                if old_prefix is None:
                    os.environ.pop("QPADM_ALLOWED_PATH_PREFIXES", None)
                else:
                    os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = old_prefix
                if old_root is None:
                    os.environ.pop("QPADM_JOBS_ROOT", None)
                else:
                    os.environ["QPADM_JOBS_ROOT"] = old_root


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

    def test_packed_binary_layout(self):
        """PACKEDANCESTRYMAP: 2 bits per genotype, GENO header, high-order first."""
        with tempfile.TemporaryDirectory() as tmp:
            # 5 individuals in 3 pops, 3 SNPs; keep pop P (indices 0, 2, 4)
            ind = os.path.join(tmp, "a.ind")
            with open(ind, "w", encoding="utf-8", newline="\n") as f:
                f.write("a U P\nb U Q\nc U P\nd U Q\ne U P\n")
            snp = os.path.join(tmp, "a.snp")
            with open(snp, "w", encoding="utf-8", newline="\n") as f:
                f.write("s1 1 0 0 A G\ns2 1 0 0 C T\ns3 1 0 0 A C\n")

            rlen = 48  # max(48, ceil(5/4)) = 48
            # Build header
            header = b"GENO       5       3 0 0"
            header = header + b"\x00" * (rlen - len(header))

            # SNP 0: individuals [0,1,2,3,4] = [0,1,2,3,1]
            # ind k -> byte k//4, shift (3-k%4)*2
            #   ind0: 0<<6=0x00, ind1: 1<<4=0x10, ind2: 2<<2=0x08, ind3: 3<<0=0x03
            #   byte0 = 0x1B, ind4: 1<<6=0x40, byte1 = 0x40
            snp0 = bytearray(rlen)
            snp0[0] = 0x1B
            snp0[1] = 0x40

            # SNP 1: [2,0,1,2,0]
            #   ind0: 2<<6=0x80, ind1: 0<<4=0, ind2: 1<<2=0x04, ind3: 2<<0=0x02
            #   byte0 = 0x86, ind4: 0<<6=0, byte1 = 0x00
            snp1 = bytearray(rlen)
            snp1[0] = 0x86

            # SNP 2: [1,2,0,1,3(missing)]
            #   ind0: 1<<6=0x40, ind1: 2<<4=0x20, ind2: 0<<2=0, ind3: 1<<0=0x01
            #   byte0 = 0x61, ind4: 3<<6=0xC0, byte1 = 0xC0
            snp2 = bytearray(rlen)
            snp2[0] = 0x61
            snp2[1] = 0xC0

            geno = os.path.join(tmp, "a.geno")
            with open(geno, "wb") as f:
                f.write(header + bytes(snp0) + bytes(snp1) + bytes(snp2))

            out = os.path.join(tmp, "out")
            g2, s2, i2 = subset_eigenstrat_by_pops(geno, snp, ind, {"P"}, out)
            self.assertTrue(os.path.isfile(g2))
            with open(g2, "rb") as f:
                rows = f.read().splitlines()
            # kept inds [0,2,4]: SNP0=[0,2,1], SNP1=[2,1,0], SNP2=[1,0,3→9]
            self.assertEqual(rows, [b"021", b"210", b"109"])
            with open(i2, "r", encoding="utf-8") as f:
                kept_lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(kept_lines), 3)
            self.assertIn("P", kept_lines[0])


if __name__ == "__main__":
    unittest.main()
