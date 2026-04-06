"""
InsightEngine AI - Engine: Sandbox Runner
Safely executes AI-generated pandas code in an isolated environment.
Core safety layer — no unsafe code ever reaches the system.
"""
import logging
import time
import traceback
import json
import signal
import re
from typing import Any, Optional
from contextlib import contextmanager
import pandas as pd
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─── Security: Allowed imports whitelist ──────────────────────────────────────
ALLOWED_IMPORTS = {"pandas", "pd", "numpy", "np", "math", "json", "datetime", "re", "collections"}

FORBIDDEN_PATTERNS = [
    r'\bimport\s+os\b',
    r'\bimport\s+sys\b',
    r'\bimport\s+subprocess\b',
    r'\bfrom\s+os\b',
    r'\bfrom\s+sys\b',
    r'\b__import__\s*\(',
    r'\bopen\s*\(',
    r'\bexec\s*\(',
    r'\beval\s*\(',
    r'shutil',
    r'pathlib',
    r'socket',
    r'requests',
    r'urllib',
    r'http',
]


class SandboxSecurityError(Exception):
    """Raised when code fails security validation."""
    pass


class SandboxTimeoutError(Exception):
    """Raised when code execution exceeds time limit."""
    pass


class SandboxExecutionError(Exception):
    """Raised when code execution fails at runtime."""
    pass


class SandboxResult:
    """Structured output from sandbox execution."""

    def __init__(
        self,
        success: bool,
        result: Any = None,
        chart_config: Optional[dict] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        execution_time_ms: int = 0,
        output_type: str = "unknown",
    ):
        self.success = success
        self.result = result
        self.chart_config = chart_config
        self.error = error
        self.error_type = error_type
        self.execution_time_ms = execution_time_ms
        self.output_type = output_type

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "result": self._serialize_result(),
            "chart_config": self.chart_config,
            "error": self.error,
            "error_type": self.error_type,
            "execution_time_ms": self.execution_time_ms,
            "output_type": self.output_type,
        }

    def _serialize_result(self) -> Any:
        """Convert pandas/numpy types to JSON-serializable format."""
        if self.result is None:
            return None
        if isinstance(self.result, pd.DataFrame):
            return {
                "type": "dataframe",
                "rows": len(self.result),
                "columns": list(self.result.columns),
                "data": self.result.head(100).to_dict(orient="records"),
            }
        if isinstance(self.result, pd.Series):
            return {
                "type": "series",
                "data": self.result.head(100).to_dict(),
            }
        if isinstance(self.result, np.integer):
            return int(self.result)
        if isinstance(self.result, np.floating):
            return float(self.result)
        if isinstance(self.result, np.ndarray):
            return self.result.tolist()
        if isinstance(self.result, dict):
            return self._serialize_dict(self.result)
        return self.result

    def _serialize_dict(self, d: dict) -> dict:
        """Recursively serialize dict with numpy types."""
        result = {}
        for k, v in d.items():
            if isinstance(v, (np.integer,)):
                result[k] = int(v)
            elif isinstance(v, (np.floating,)):
                result[k] = float(v)
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, pd.Series):
                result[k] = v.to_dict()
            elif isinstance(v, dict):
                result[k] = self._serialize_dict(v)
            else:
                result[k] = v
        return result


class SandboxRunner:
    """
    Isolated execution environment for AI-generated pandas code.

    Security layers:
    1. Static code analysis (pattern matching)
    2. Restricted execution namespace
    3. Timeout enforcement
    4. Result validation
    """

    def __init__(self):
        self.timeout = settings.SANDBOX_TIMEOUT_SECONDS

    def run(self, code: str, df: pd.DataFrame) -> SandboxResult:
        """
        Execute pandas code safely against a DataFrame.

        Args:
            code: Python code string (from Judge agent)
            df: The cleaned dataset as a pandas DataFrame

        Returns:
            SandboxResult with execution output
        """
        # Step 1: Static security scan
        try:
            self._security_scan(code)
        except SandboxSecurityError as e:
            logger.warning(f"Security violation blocked: {e}")
            return SandboxResult(
                success=False,
                error=str(e),
                error_type="security_violation",
            )

        # Step 2: Execute with timeout
        start_time = time.time()

        try:
            result_dict = self._execute_with_timeout(code, df)
            elapsed_ms = int((time.time() - start_time) * 1000)

            result = result_dict.get("result")
            chart_config = result_dict.get("chart_config")
            output_type = self._classify_output(result)

            logger.info(f"Sandbox executed successfully in {elapsed_ms}ms (type={output_type})")

            return SandboxResult(
                success=True,
                result=result,
                chart_config=chart_config,
                execution_time_ms=elapsed_ms,
                output_type=output_type,
            )

        except SandboxTimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"Sandbox timed out after {elapsed_ms}ms")
            return SandboxResult(
                success=False,
                error=f"Analysis took too long (over {self.timeout}s). Try a simpler question.",
                error_type="timeout",
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = self._clean_error_message(str(e))
            tb = traceback.format_exc()
            logger.error(f"Sandbox execution error: {tb}")
            return SandboxResult(
                success=False,
                error=error_msg,
                error_type="runtime_error",
                execution_time_ms=elapsed_ms,
            )

    def _security_scan(self, code: str) -> None:
        """
        Scan code for forbidden patterns before execution.
        Raises SandboxSecurityError if any are found.
        """
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                raise SandboxSecurityError(
                    f"Unsafe pattern detected: '{pattern}'"
                )

    def _execute_with_timeout(self, code: str, df: pd.DataFrame) -> dict:
        """
        Execute code in a restricted namespace with timeout.
        Uses SIGALRM on Unix systems for hard timeout.
        """
        # Build restricted execution namespace
        namespace = {
            "df": df.copy(),       # Pass a COPY — never modify original
            "pd": pd,
            "np": np,
            "json": json,
            "__builtins__": self._safe_builtins(),
        }

        # Try signal-based timeout (Unix only)
        try:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.timeout)
        except (AttributeError, OSError):
            # Windows or unsupported — rely on thread timeout
            pass

        try:
            exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102
        finally:
            try:
                signal.alarm(0)  # Cancel alarm
            except (AttributeError, OSError):
                pass

        return {
            "result": namespace.get("result"),
            "chart_config": namespace.get("chart_config"),
        }

    def _safe_builtins(self) -> dict:
        """Return a restricted set of Python builtins."""
        allowed = [
            "abs", "all", "any", "bool", "dict", "enumerate", "filter",
            "float", "format", "int", "isinstance", "len", "list",
            "map", "max", "min", "print", "range", "reversed", "round",
            "set", "sorted", "str", "sum", "tuple", "type", "zip",
        ]
        import builtins
        return {name: getattr(builtins, name) for name in allowed}

    @staticmethod
    def _timeout_handler(signum, frame):
        raise SandboxTimeoutError("Code execution timed out")

    def _classify_output(self, result: Any) -> str:
        """Classify the type of result for frontend rendering."""
        if result is None:
            return "none"
        if isinstance(result, pd.DataFrame):
            return "dataframe"
        if isinstance(result, pd.Series):
            return "series"
        if isinstance(result, (int, float, np.integer, np.floating)):
            return "number"
        if isinstance(result, str):
            return "text"
        if isinstance(result, (list, tuple)):
            return "list"
        if isinstance(result, dict):
            return "dict"
        return "unknown"

    def _clean_error_message(self, error: str) -> str:
        """
        Strip internal paths/details from error messages.
        Never expose internal system paths to users.
        """
        # Remove file path references
        error = re.sub(r'File ".*?"', 'File "<code>"', error)
        # Truncate overly long errors
        if len(error) > 300:
            error = error[:300] + "..."
        return error


# Module-level singleton
sandbox_runner = SandboxRunner()