"""Unit tests for qpAdm API helpers (no qpAdm binary)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
