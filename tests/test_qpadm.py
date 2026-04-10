"""Unit tests for qpAdm ADMIXTOOLS 2 helpers (no R/Rscript required)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from qpadm.paths import resolve_under_allowed
from qpadm.workdir import validate_pop_token, save_request, write_job_workdir, REQUEST_NAME


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


class TestSaveRequest(unittest.TestCase):
    def test_writes_request_json_with_f2_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("x.geno", "x.snp", "x.ind"):
                open(os.path.join(tmp, name), "w").close()
            old_prefix = os.environ.get("QPADM_ALLOWED_PATH_PREFIXES")
            old_f2 = os.environ.get("QPADM_F2_DIR")
            os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = tmp
            os.environ["QPADM_F2_DIR"] = "/data/qpadm_f2"
            try:
                work = os.path.join(tmp, "job_work")
                snap = save_request(
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
                self.assertIn("geno_prefix", snap)
                self.assertTrue(snap["geno_prefix"].endswith("x"))
                self.assertEqual(snap["f2_dir"], "/data/qpadm_f2")
                self.assertEqual(snap["details"], True)
                self.assertNotIn("ind_mode", snap)

                with open(os.path.join(work, REQUEST_NAME), encoding="utf-8") as f:
                    on_disk = json.load(f)
                self.assertEqual(on_disk["left_pops"], ["Target.Pop", "Source.Pop"])
                self.assertEqual(on_disk["right_pops"], ["Mbuti.DG"])
                self.assertIn("f2_dir", on_disk)
            finally:
                if old_prefix is None:
                    os.environ.pop("QPADM_ALLOWED_PATH_PREFIXES", None)
                else:
                    os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = old_prefix
                if old_f2 is None:
                    os.environ.pop("QPADM_F2_DIR", None)
                else:
                    os.environ["QPADM_F2_DIR"] = old_f2

    def test_no_f2_dir_omits_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("x.geno", "x.snp", "x.ind"):
                open(os.path.join(tmp, name), "w").close()
            old_prefix = os.environ.get("QPADM_ALLOWED_PATH_PREFIXES")
            old_f2 = os.environ.get("QPADM_F2_DIR")
            os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = tmp
            os.environ.pop("QPADM_F2_DIR", None)
            try:
                work = os.path.join(tmp, "job_work")
                snap = save_request(
                    work,
                    {
                        "left_pops": ["A"],
                        "right_pops": ["B"],
                        "genotypename": os.path.join(tmp, "x.geno"),
                        "snpname": os.path.join(tmp, "x.snp"),
                        "indivname": os.path.join(tmp, "x.ind"),
                        "allsnps": True,
                        "inbreed": False,
                        "details": False,
                    },
                )
                self.assertNotIn("f2_dir", snap)
                self.assertIn("geno_prefix", snap)
            finally:
                if old_prefix is None:
                    os.environ.pop("QPADM_ALLOWED_PATH_PREFIXES", None)
                else:
                    os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = old_prefix
                if old_f2 is None:
                    os.environ.pop("QPADM_F2_DIR", None)
                else:
                    os.environ["QPADM_F2_DIR"] = old_f2

    def test_default_triplet_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("d.geno", "d.snp", "d.ind"):
                open(os.path.join(tmp, name), "w").close()
            env_overrides = {
                "QPADM_ALLOWED_PATH_PREFIXES": tmp,
                "QPADM_DEFAULT_GENO": os.path.join(tmp, "d.geno"),
                "QPADM_DEFAULT_SNP": os.path.join(tmp, "d.snp"),
                "QPADM_DEFAULT_IND": os.path.join(tmp, "d.ind"),
            }
            old_vals = {}
            for k, v in env_overrides.items():
                old_vals[k] = os.environ.get(k)
                os.environ[k] = v
            os.environ.pop("QPADM_F2_DIR", None)
            try:
                work = os.path.join(tmp, "job_work")
                snap = save_request(
                    work,
                    {
                        "left_pops": ["X"],
                        "right_pops": ["Y"],
                        "allsnps": False,
                        "inbreed": False,
                        "details": False,
                    },
                )
                self.assertTrue(snap["geno_prefix"].endswith("d"))
            finally:
                for k, v in old_vals.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


class TestWriteJobWorkdir(unittest.TestCase):
    def test_is_alias_for_save_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("x.geno", "x.snp", "x.ind"):
                open(os.path.join(tmp, name), "w").close()
            old = os.environ.get("QPADM_ALLOWED_PATH_PREFIXES")
            os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = tmp
            os.environ.pop("QPADM_F2_DIR", None)
            try:
                work = os.path.join(tmp, "job_work")
                snap = write_job_workdir(
                    work,
                    {
                        "left_pops": ["A", "B"],
                        "right_pops": ["C"],
                        "genotypename": os.path.join(tmp, "x.geno"),
                        "snpname": os.path.join(tmp, "x.snp"),
                        "indivname": os.path.join(tmp, "x.ind"),
                        "allsnps": False,
                        "inbreed": False,
                        "details": False,
                    },
                )
                self.assertIn("geno_prefix", snap)
                self.assertTrue(os.path.isfile(os.path.join(work, REQUEST_NAME)))
            finally:
                if old is None:
                    os.environ.pop("QPADM_ALLOWED_PATH_PREFIXES", None)
                else:
                    os.environ["QPADM_ALLOWED_PATH_PREFIXES"] = old


class TestRunnerJsonParsing(unittest.TestCase):
    """Test that the runner correctly handles JSON output from the R script."""

    def _make_job_env(self, tmp, request_data):
        """Helper: create a minimal job dir with request.json."""
        job_id = "test-job-1"
        job_dir = os.path.join(tmp, job_id, "work")
        os.makedirs(job_dir)
        with open(os.path.join(job_dir, REQUEST_NAME), "w") as f:
            json.dump(request_data, f)
        return job_id

    @patch("qpadm.runner.store")
    @patch("qpadm.runner.subprocess.run")
    def test_successful_run_stores_structured_result(self, mock_run, mock_store):
        with tempfile.TemporaryDirectory() as tmp:
            mock_store.jobs_root.return_value = tmp
            mock_store.get_job.return_value = {"status": "queued"}

            job_id = self._make_job_env(tmp, {
                "left_pops": ["Target", "Source"],
                "right_pops": ["Out1"],
                "geno_prefix": "/fake/ref",
                "allsnps": False,
                "details": False,
            })

            r_output = {
                "weights": [
                    {"target": "Target", "left": "Source", "weight": 1.0, "se": 0.0, "z": 999.0}
                ],
                "rankdrop": [
                    {"f4rank": 0, "dof": 0, "chisq": 0.0, "p": 1.0}
                ],
                "elapsed_sec": 1.234,
            }
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = json.dumps(r_output)
            mock_proc.stderr = "qpadm() completed in 1.23 seconds\n"
            mock_run.return_value = mock_proc

            from qpadm.runner import run_qp_adm_job
            run_qp_adm_job(job_id)

            mock_store.update_job.assert_called()
            last_call = mock_store.update_job.call_args_list[-1]
            self.assertEqual(last_call.args[1], "done")
            result = last_call.kwargs.get("result") or last_call.args[2]
            self.assertIn("weights", result)
            self.assertEqual(result["weights"][0]["weight"], 1.0)

    @patch("qpadm.runner.store")
    @patch("qpadm.runner.subprocess.run")
    def test_failed_run_captures_error(self, mock_run, mock_store):
        with tempfile.TemporaryDirectory() as tmp:
            mock_store.jobs_root.return_value = tmp
            mock_store.get_job.return_value = {"status": "queued"}

            job_id = self._make_job_env(tmp, {
                "left_pops": ["Target", "Source"],
                "right_pops": ["Out1"],
                "geno_prefix": "/fake/ref",
                "allsnps": False,
                "details": False,
            })

            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.stdout = json.dumps({"error": "pop X not found in f2 dir"})
            mock_proc.stderr = "Error in qpadm()\n"
            mock_run.return_value = mock_proc

            from qpadm.runner import run_qp_adm_job
            run_qp_adm_job(job_id)

            last_call = mock_store.update_job.call_args_list[-1]
            self.assertEqual(last_call.args[1], "failed")
            error_str = last_call.kwargs.get("error") or last_call.args[2]
            self.assertIn("pop X not found", error_str)


if __name__ == "__main__":
    unittest.main()
