"""Unit tests for qpAdm job helpers (no ADMIXTOOLS binary required)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from qpadm.materialize import DEFAULT_SOURCES_MANIFEST, materialize_pop_list_files
from qpadm.runner import validate_par_filename


class TestValidateParFilename(unittest.TestCase):
    def test_accepts_basename(self):
        self.assertEqual(validate_par_filename("qpAdm.par"), "qpAdm.par")
        self.assertEqual(validate_par_filename("  my.run.par  "), "my.run.par")

    def test_rejects_invalid_names(self):
        with self.assertRaises(ValueError):
            validate_par_filename("no spaces.par")
        with self.assertRaises(ValueError):
            validate_par_filename(".hidden.par")

    def test_strips_directories(self):
        """Only the basename is used; path segments in the form field are ignored."""
        self.assertEqual(validate_par_filename("subdir/qpAdm.par"), "qpAdm.par")
        self.assertEqual(validate_par_filename("../qpAdm.par"), "qpAdm.par")


class TestMaterializePrecedence(unittest.TestCase):
    def test_manifest_overwrites_zip_stale_list(self):
        with tempfile.TemporaryDirectory() as work:
            par = os.path.join(work, "qpAdm.par")
            with open(par, "w", encoding="utf-8") as f:
                f.write("popleft: MyPop.AG\n")
            token_path = os.path.join(work, "MyPop.AG")
            with open(token_path, "w", encoding="utf-8") as f:
                f.write("stale_id\n")
            man = os.path.join(work, DEFAULT_SOURCES_MANIFEST)
            with open(man, "w", encoding="utf-8") as f:
                json.dump({"MyPop.AG": ["a", "b"]}, f)
            materialize_pop_list_files(
                work,
                par,
                manifest_basename=DEFAULT_SOURCES_MANIFEST,
                auto_from_ind=True,
            )
            with open(token_path, encoding="utf-8") as f:
                body = f.read()
            self.assertEqual(body, "a\nb\n")

    def test_ind_expands_when_no_manifest(self):
        with tempfile.TemporaryDirectory() as work:
            ind_rel = os.path.join("refs", "tiny.ind")
            ind_abs = os.path.join(work, ind_rel)
            os.makedirs(os.path.dirname(ind_abs), exist_ok=True)
            with open(ind_abs, "w", encoding="utf-8") as f:
                f.write("sample1 U Target.Pop\n")
            par = os.path.join(work, "qpAdm.par")
            with open(par, "w", encoding="utf-8") as f:
                f.write(f"indivname: {ind_rel}\n")
                f.write("popleft: Target.Pop\n")
            materialize_pop_list_files(
                work,
                par,
                manifest_basename=DEFAULT_SOURCES_MANIFEST,
                auto_from_ind=True,
            )
            p = os.path.join(work, "Target.Pop")
            self.assertTrue(os.path.isfile(p))
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read().strip(), "sample1")


if __name__ == "__main__":
    unittest.main()
