"""Unit tests for ADMIXTURE job helpers (no admixture binary required)."""

from __future__ import annotations

import unittest

from admixture_jobs.runner import validate_plink_prefix


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


if __name__ == "__main__":
    unittest.main()
