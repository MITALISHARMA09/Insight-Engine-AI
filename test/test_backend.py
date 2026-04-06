"""
InsightEngine AI - Test Suite
Tests for: sandbox, profiler, cleaning executor, schemas, and agent logic.
Run with: pytest tests/ -v
"""
import sys
import os
import json
import pandas as pd
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """A realistic sample DataFrame for testing."""
    return pd.DataFrame({
        "employee_id": range(1, 11),
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve",
                  "Frank", "Grace", "Hank", "Ivy", "Jack"],
        "department": ["Sales", "HR", "Sales", "IT", "HR",
                       "Sales", "IT", "HR", "Sales", "IT"],
        "salary": [55000, 48000, 62000, 75000, 44000,
                   58000, 80000, 46000, None, 72000],
        "age": [28, 35, 42, 31, 29, 38, 45, 33, 27, 40],
        "hire_date": ["2020-01-15", "2018-06-22", "2019-03-10",
                      "2021-07-01", "2022-01-05", "2017-11-30",
                      "2016-08-14", "2020-09-25", "2023-02-28", "2015-04-12"],
    })


@pytest.fixture
def dirty_df():
    """DataFrame with intentional quality issues."""
    return pd.DataFrame({
        "id":             [1,    2,    2,    3,    4,    5],
        "value":          [10.0, None, None, 30.0, 30.0, 9999.0],  # 2 missing, 1 outlier
        "category":       ["A",  "B",  "B",  "A",  None, "C"],
        "number_as_str":  ["100","200","200","400", "500","600"],   # true dupes in rows 1/2
    })
    # Row 1 and Row 2 are now exact duplicates → 1 duplicate detected


# ─── Sandbox Tests ────────────────────────────────────────────────────────────

class TestSandboxRunner:
    def setup_method(self):
        from app.engine.sandbox_runner import SandboxRunner
        self.sandbox = SandboxRunner()

    def test_simple_calculation(self, sample_df):
        code = "result = df['salary'].mean()"
        res = self.sandbox.run(code, sample_df)
        assert res.success is True
        assert res.result is not None
        assert isinstance(res.result, float)

    def test_groupby_operation(self, sample_df):
        code = """
result = df.groupby('department')['salary'].mean().to_dict()
"""
        res = self.sandbox.run(code, sample_df)
        assert res.success is True
        assert isinstance(res.result, dict)
        assert "Sales" in res.result or "HR" in res.result

    def test_blocks_os_import(self, sample_df):
        code = "import os\nresult = os.getcwd()"
        res = self.sandbox.run(code, sample_df)
        assert res.success is False
        assert res.error_type == "security_violation"

    def test_blocks_eval(self, sample_df):
        code = "result = eval('1+1')"
        res = self.sandbox.run(code, sample_df)
        assert res.success is False

    def test_blocks_open_file(self, sample_df):
        code = "result = open('/etc/passwd').read()"
        res = self.sandbox.run(code, sample_df)
        assert res.success is False

    def test_dataframe_result(self, sample_df):
        code = "result = df[df['department'] == 'Sales']"
        res = self.sandbox.run(code, sample_df)
        assert res.success is True
        assert res.output_type == "dataframe"

    def test_does_not_modify_original(self, sample_df):
        original_len = len(sample_df)
        code = "df.drop(0, inplace=True)\nresult = len(df)"
        self.sandbox.run(code, sample_df)
        assert len(sample_df) == original_len  # original unchanged

    def test_serialization_numpy_int(self, sample_df):
        code = "result = df['age'].max()"
        res = self.sandbox.run(code, sample_df)
        assert res.success is True
        serialized = res.to_dict()
        # Should be JSON-serializable
        json.dumps(serialized)

    def test_empty_result(self, sample_df):
        code = "result = None"
        res = self.sandbox.run(code, sample_df)
        assert res.success is True
        assert res.result is None


# ─── Profiler Tests ───────────────────────────────────────────────────────────

class TestDataProfiler:
    def setup_method(self):
        from app.cleaning.profiler import DataProfiler
        self.profiler = DataProfiler()

    def test_detects_missing_values(self, dirty_df):
        report = self.profiler.profile(dirty_df)
        assert "value" in report["missing_values"]
        assert report["missing_values"]["value"]["count"] == 2

    def test_detects_duplicates(self, dirty_df):
        report = self.profiler.profile(dirty_df)
        assert report["duplicates"]["has_duplicates"] is True
        assert report["duplicates"]["count"] >= 1

    def test_detects_outliers(self):
        # Need ≥10 values and outlier ratio >0.1% for detection
        normal = [10, 12, 11, 13, 10, 12, 9, 11, 13, 10]
        df = pd.DataFrame({"x": normal + [9999, 9998]})
        report = self.profiler.profile(df)
        assert "x" in report["outliers"]

    def test_quality_score_clean_data(self, sample_df):
        # Remove the one missing value
        df = sample_df.dropna()
        report = self.profiler.profile(df)
        assert report["quality_score"] > 80.0

    def test_quality_score_dirty_data(self, dirty_df):
        report = self.profiler.profile(dirty_df)
        assert report["quality_score"] < 85.0

    def test_friendly_summary_no_issues(self, sample_df):
        df = sample_df.fillna(0)
        report = self.profiler.profile(df)
        summary = self.profiler.user_friendly_summary(report)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_report_has_required_keys(self, sample_df):
        report = self.profiler.profile(sample_df)
        required = ["total_rows", "total_columns", "missing_values",
                    "duplicates", "outliers", "quality_score"]
        for key in required:
            assert key in report


# ─── Cleaning Executor Tests ──────────────────────────────────────────────────

class TestCleaningExecutor:
    def setup_method(self):
        from app.cleaning.executor import CleaningExecutor
        self.executor = CleaningExecutor()

    def test_fill_missing_median(self, dirty_df):
        plan = [{"action": "fill_missing", "column": "value", "method": "median"}]
        df_clean, report = self.executor.execute(dirty_df, plan)
        assert df_clean["value"].isnull().sum() == 0
        assert len(report["actions_applied"]) == 1

    def test_fill_missing_mode(self, dirty_df):
        plan = [{"action": "fill_missing", "column": "category", "method": "mode"}]
        df_clean, report = self.executor.execute(dirty_df, plan)
        assert df_clean["category"].isnull().sum() == 0

    def test_remove_duplicates(self, dirty_df):
        original_len = len(dirty_df)
        plan = [{"action": "remove_duplicates"}]
        df_clean, report = self.executor.execute(dirty_df, plan)
        assert len(df_clean) < original_len

    def test_fix_dtype_numeric(self):
        df = pd.DataFrame({"amount": ["100", "200", "300"]})
        plan = [{"action": "fix_dtype", "column": "amount", "target_type": "numeric"}]
        df_clean, _ = self.executor.execute(df, plan)
        assert pd.api.types.is_numeric_dtype(df_clean["amount"])

    def test_cap_outliers(self, dirty_df):
        plan = [{"action": "cap_outliers", "column": "value", "upper": 100}]
        df_clean, _ = self.executor.execute(dirty_df, plan)
        assert df_clean["value"].max() <= 100

    def test_does_not_modify_original(self, dirty_df):
        original_vals = dirty_df["value"].copy()
        plan = [{"action": "fill_missing", "column": "value", "method": "zero"}]
        self.executor.execute(dirty_df, plan)
        pd.testing.assert_series_equal(dirty_df["value"], original_vals)

    def test_skips_invalid_action(self, sample_df):
        plan = [{"action": "nonexistent_action", "column": "salary"}]
        df_clean, report = self.executor.execute(sample_df, plan)
        assert len(report["actions_applied"]) == 0

    def test_skips_missing_column(self, sample_df):
        plan = [{"action": "fill_missing", "column": "nonexistent_col", "method": "median"}]
        df_clean, report = self.executor.execute(sample_df, plan)
        assert len(report["actions_skipped"]) == 1

    def test_empty_plan(self, sample_df):
        df_clean, report = self.executor.execute(sample_df, [])
        assert len(df_clean) == len(sample_df)
        assert len(report["actions_applied"]) == 0


# ─── Coder Agent Safety Tests ─────────────────────────────────────────────────

class TestCoderAgentSafety:
    def setup_method(self):
        from app.agents.code_coder_a import CoderAAgent
        self.agent = CoderAAgent()

    def test_safe_code_passes(self):
        code = "result = df['salary'].mean()"
        is_safe, reason = self.agent.validate_code_safety(code)
        assert is_safe is True

    def test_os_import_fails(self):
        code = "import os\nresult = os.listdir('.')"
        is_safe, reason = self.agent.validate_code_safety(code)
        assert is_safe is False

    def test_subprocess_fails(self):
        code = "import subprocess\nresult = subprocess.run(['ls'])"
        is_safe, reason = self.agent.validate_code_safety(code)
        assert is_safe is False

    def test_eval_fails(self):
        code = "result = eval('print(1)')"
        is_safe, reason = self.agent.validate_code_safety(code)
        assert is_safe is False


# ─── Pydantic Schema Tests ────────────────────────────────────────────────────

class TestSchemas:
    def test_query_request_valid(self):
        from app.api.schemas import QueryRequest
        req = QueryRequest(question="What is the average salary?")
        assert req.question == "What is the average salary?"

    def test_query_request_strips_whitespace(self):
        from app.api.schemas import QueryRequest
        req = QueryRequest(question="  What is the average salary?  ")
        assert req.question == "What is the average salary?"

    def test_query_request_too_short(self):
        from app.api.schemas import QueryRequest
        with pytest.raises(Exception):
            QueryRequest(question="hi")

    def test_upload_response_model(self):
        from app.api.schemas import UploadResponse
        resp = UploadResponse(
            dataset_id="abc-123",
            original_filename="data.csv",
            status="ready",
            row_count=1000,
            column_count=5,
            columns=["a", "b", "c", "d", "e"],
            domain="sales",
            quality_score=87.5,
            quality_summary="Good quality",
            has_embeddings=True,
        )
        assert resp.dataset_id == "abc-123"
        assert resp.status == "ready"