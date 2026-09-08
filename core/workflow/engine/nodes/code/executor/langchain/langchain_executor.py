import os
import tempfile
from typing import Any

from langchain_sandbox import PyodideSandbox

from workflow.configs.app_config import (
    DEFAULT_CODE_EXEC_MEMORY_LIMIT_MB,
    DEFAULT_CODE_EXEC_TIMEOUT_SEC,
    MAX_CODE_EXEC_MEMORY_LIMIT_MB,
    MAX_CODE_EXEC_TIMEOUT_SEC,
    MIN_CODE_EXEC_MEMORY_LIMIT_MB,
    MIN_CODE_EXEC_TIMEOUT_SEC,
)
from workflow.engine.nodes.code.executor.base_executor import BaseExecutor
from workflow.exception.e import CustomException
from workflow.exception.errors.err_code import CodeEnum
from workflow.extensions.otlp.trace.span import Span

MAX_ERROR_MESSAGE_LENGTH = 4096


class _FilePyodideSandbox(PyodideSandbox):
    """Use the official CLI file input without rewriting Python escapes.

    langchain-sandbox 0.0.6 only exposes inline code through execute(), and
    @langchain/pyodide-sandbox 0.0.4 replaces literal backslash-n sequences in
    that input. Its official -f entry point reads source verbatim instead.
    Keep the SDK's command, process lifecycle and resource limits, adapting
    only the input flag until the SDK exposes file input publicly.
    """

    def __init__(self, source_path: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._source_path = source_path

    def _build_command(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        memory_limit_mb: int | None = None,
    ) -> list[str]:
        command = super()._build_command(
            code,
            session_bytes=session_bytes,
            session_metadata=session_metadata,
            memory_limit_mb=memory_limit_mb,
        )
        code_flag_index = command.index("-c")
        command[code_flag_index : code_flag_index + 2] = ["-f", self._source_path]
        return command


class LangchainExecutor(BaseExecutor):
    """
    Code executor using Langchain Pyodide sandbox.

    Executes Python code in a browser-based sandbox environment using Pyodide,
    providing isolation and security through the Langchain sandbox implementation.
    """

    async def execute(
        self, language: str, code: str, timeout: int, span: Span, **kwargs: Any
    ) -> str:
        """
        Execute code using Langchain Pyodide sandbox.

        :param language: Programming language (currently only python supported)
        :param code: Code string to execute
        :param timeout: Maximum execution time in seconds
        :param span: Tracing span for logging
        :param kwargs: Additional execution parameters
        :return: Execution result as string
        :raises CustomException: If code execution fails
        """
        try:
            bounded_timeout = _bounded_timeout(timeout)
            bounded_memory_limit = _bounded_memory_limit()
            # Grant Deno read access only to this invocation's source file in
            # addition to the SDK's existing node_modules permissions. File
            # input preserves escapes, import discovery and top-level await.
            # The context manager removes the file on success or failure.
            with tempfile.TemporaryDirectory(prefix="astron-code-") as source_dir:
                source_path = os.path.realpath(os.path.join(source_dir, "source.py"))
                # Close the writer before Deno opens the file (also on Windows).
                with open(
                    source_path, "w", encoding="utf-8", newline=""
                ) as source_file:
                    source_file.write(code)
                sandbox = _FilePyodideSandbox(
                    source_path,
                    allow_env=False,
                    allow_read=["node_modules", source_path],
                    allow_write=False,
                    allow_net=False,
                    allow_run=False,
                    allow_ffi=False,
                )
                result = await sandbox.execute(
                    code,
                    timeout_seconds=bounded_timeout,
                    memory_limit_mb=bounded_memory_limit,
                )
            if result.status == "success":
                return result.stdout if result.stdout else ""
            error_message = (result.stderr or "Code execution failed").strip()
            raise CustomException(
                err_code=(
                    CodeEnum.CODE_EXECUTION_TIMEOUT_ERROR
                    if _is_timeout_error(error_message)
                    else CodeEnum.CODE_EXECUTION_ERROR
                ),
                err_msg=error_message[:MAX_ERROR_MESSAGE_LENGTH],
            )

        except CustomException as e:
            raise e

        except Exception as e:
            raise CustomException(
                err_code=CodeEnum.CODE_EXECUTION_ERROR,
                cause_error=e,
            ) from e


def _bounded_timeout(timeout: int) -> int:
    """Normalize a requested timeout to the safe Pyodide execution range."""
    try:
        requested_timeout = int(timeout)
    except (TypeError, ValueError):
        requested_timeout = DEFAULT_CODE_EXEC_TIMEOUT_SEC
    return max(
        MIN_CODE_EXEC_TIMEOUT_SEC,
        min(requested_timeout, MAX_CODE_EXEC_TIMEOUT_SEC),
    )


def _bounded_memory_limit() -> int:
    """Read and clamp the optional Pyodide V8 heap limit."""
    # The configuration model validates this value at startup.  Reading the
    # environment here as well keeps the executor robust in tests and when a
    # process reloads configuration without rebuilding the settings object.
    try:
        requested_limit = int(
            os.getenv(
                "CODE_EXEC_MEMORY_LIMIT_MB", str(DEFAULT_CODE_EXEC_MEMORY_LIMIT_MB)
            )
        )
    except (TypeError, ValueError):
        requested_limit = DEFAULT_CODE_EXEC_MEMORY_LIMIT_MB
    return max(
        MIN_CODE_EXEC_MEMORY_LIMIT_MB,
        min(requested_limit, MAX_CODE_EXEC_MEMORY_LIMIT_MB),
    )


def _is_timeout_error(message: str) -> bool:
    """Recognize timeout diagnostics emitted by the Pyodide wrapper."""
    normalized = message.lower()
    return "timed out" in normalized or "timeout" in normalized
