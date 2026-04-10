"""Unit tests for ADMIXTURE job helpers (no admixture binary required)."""

from __future__ import annotations

import os
import tempfile
import unittest

from admixture_jobs.runner import resolve_host_plink_bed, validate_plink_prefix


class TestValidatePlinkPrefix(unittest.TestCase):
    def test_accepts_basename(self):
        self.assertEqual(validate_plink_prefix("v62.0_HO_small"), "v62.0_HO_small")
        self.assertEqual(validate_plink_prefix("  mydata  "), "mydata")

    def test_rejects_invalid_names(self):
        with self.assertRaises(ValueError):
            validate_plink_prefix("no spaces")
        with self.assertRaises(ValueError):
            validate_plink_prefix(".hidden")

    def test_strips_directories(self):
        self.assertEqual(
            validate_plink_prefix("subdir/v62.0_HO_small"),
            "v62.0_HO_small",
        )
        self.assertEqual(
            validate_plink_prefix("../v62.0_HO_small"),
            "v62.0_HO_small",
        )


class TestResolveHostPlinkBed(unittest.TestCase):
    def test_resolves_when_triplet_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ext in ("bed", "bim", "fam"):
                path = os.path.join(tmp, f"mydata.{ext}")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("x" if ext != "bed" else "\0")
            old = os.environ.get("ADMIXTURE_HOST_PLINK_ROOT")
            os.environ["ADMIXTURE_HOST_PLINK_ROOT"] = tmp
            try:
                bed = resolve_host_plink_bed("mydata")
                self.assertTrue(bed.endswith("mydata.bed"))
                self.assertTrue(os.path.isfile(bed))
            finally:
                if old is None:
                    os.environ.pop("ADMIXTURE_HOST_PLINK_ROOT", None)
                else:
                    os.environ["ADMIXTURE_HOST_PLINK_ROOT"] = old

    def test_rejects_when_root_unset(self):
        old = os.environ.pop("ADMIXTURE_HOST_PLINK_ROOT", None)
        try:
            with self.assertRaises(ValueError) as ctx:
                resolve_host_plink_bed("mydata")
            self.assertIn("not configured", str(ctx.exception).lower())
        finally:
            if old is not None:
                os.environ["ADMIXTURE_HOST_PLINK_ROOT"] = old


if __name__ == "__main__":
    unittest.main()
