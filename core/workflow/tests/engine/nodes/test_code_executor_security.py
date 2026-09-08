from pathlib import Path

import pytest

from workflow.configs.app_config import (
    DEFAULT_CODE_EXEC_MEMORY_LIMIT_MB,
    DEFAULT_CODE_EXECUTOR_TYPE,
    CodeExecutorConfig,
)
from workflow.engine.nodes.code.executor.base_executor import CodeExecutorFactory
from workflow.engine.nodes.code.executor.langchain.langchain_executor import (
    MAX_ERROR_MESSAGE_LENGTH,
    LangchainExecutor,
    _bounded_memory_limit,
    _bounded_timeout,
    _is_timeout_error,
)
from workflow.engine.nodes.code.executor.local.local_executor import LocalExecutor
from workflow.exception.e import CustomException


@pytest.mark.parametrize("executor_type", ["", "disabled", "local"])
def test_unsafe_or_missing_executor_fails_closed(executor_type: str) -> None:
    with pytest.raises(CustomException, match="local executor is disabled"):
        CodeExecutorFactory.create_executor(executor_type)


def test_code_executor_configuration_defaults_to_builtin_isolated_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODE_EXEC_TYPE", raising=False)

    config = CodeExecutorConfig()
    assert config.exec_type == DEFAULT_CODE_EXECUTOR_TYPE == "langchain"
    assert config.memory_limit_mb == DEFAULT_CODE_EXEC_MEMORY_LIMIT_MB


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(0, 1), (-1, 1), (10, 10), (601, 600), (10_000, 600)],
)
def test_langchain_executor_bounds_timeout(requested: int, expected: int) -> None:
    assert _bounded_timeout(requested) == expected


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("invalid", 256), ("64", 128), ("256", 256), ("4096", 2048)],
)
def test_langchain_executor_bounds_memory(
    monkeypatch: pytest.MonkeyPatch, configured: str, expected: int
) -> None:
    monkeypatch.setenv("CODE_EXEC_MEMORY_LIMIT_MB", configured)
    assert _bounded_memory_limit() == expected


@pytest.mark.asyncio
async def test_langchain_executor_passes_isolation_and_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeResult:
        status = "success"
        stdout = '{"result":"ok"}'
        stderr = None

    class FakeSandbox:
        def __init__(self, source_path: str, **kwargs: object) -> None:
            calls["source_path"] = source_path
            assert Path(source_path).read_text() == "print('ok')"
            calls["permissions"] = kwargs

        async def execute(self, _code: str, **kwargs: object) -> FakeResult:
            calls["execute"] = kwargs
            return FakeResult()

    monkeypatch.setenv("CODE_EXEC_MEMORY_LIMIT_MB", "9999")
    monkeypatch.setattr(
        "workflow.engine.nodes.code.executor.langchain.langchain_executor._FilePyodideSandbox",
        FakeSandbox,
    )

    result = await LangchainExecutor().execute(
        language="python", code="print('ok')", timeout=9999, span=None  # type: ignore[arg-type]
    )

    assert result == '{"result":"ok"}'
    assert calls["permissions"] == {
        "allow_env": False,
        "allow_read": ["node_modules", calls["source_path"]],
        "allow_write": False,
        "allow_net": False,
        "allow_run": False,
        "allow_ffi": False,
    }
    assert calls["execute"] == {
        "timeout_seconds": 600,
        "memory_limit_mb": 2048,
    }


@pytest.mark.asyncio
async def test_langchain_executor_truncates_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        status = "error"
        stdout = None
        stderr = "x" * (MAX_ERROR_MESSAGE_LENGTH + 100)

    class FakeSandbox:
        def __init__(self, _source_path: str, **_kwargs: object) -> None:
            return None

        async def execute(self, _code: str, **_kwargs: object) -> FakeResult:
            return FakeResult()

    monkeypatch.setattr(
        "workflow.engine.nodes.code.executor.langchain.langchain_executor._FilePyodideSandbox",
        FakeSandbox,
    )

    with pytest.raises(CustomException) as exc_info:
        await LangchainExecutor().execute(
            language="python", code="raise Exception()", timeout=1, span=None  # type: ignore[arg-type]
        )

    assert len(str(exc_info.value)) <= MAX_ERROR_MESSAGE_LENGTH + 100


@pytest.mark.asyncio
async def test_langchain_executor_reports_timeout_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        status = "error"
        stdout = None
        stderr = "Execution timed out after 10 seconds"

    class FakeSandbox:
        def __init__(self, _source_path: str, **_kwargs: object) -> None:
            return None

        async def execute(self, _code: str, **_kwargs: object) -> FakeResult:
            return FakeResult()

    monkeypatch.setattr(
        "workflow.engine.nodes.code.executor.langchain.langchain_executor._FilePyodideSandbox",
        FakeSandbox,
    )

    with pytest.raises(CustomException) as exc_info:
        await LangchainExecutor().execute(
            language="python", code="while True: pass", timeout=10, span=None  # type: ignore[arg-type]
        )

    assert exc_info.value.code == 21603
    assert _is_timeout_error("Execution timed out")
    assert not _is_timeout_error("NameError: missing")


@pytest.mark.asyncio
async def test_local_executor_never_executes_submitted_code(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    submitted_code = f"open({str(marker)!r}, 'w').write('executed')"

    with pytest.raises(CustomException, match="disabled for security"):
        await LocalExecutor().execute(
            language="python",
            code=submitted_code,
            timeout=1,
            span=None,  # type: ignore[arg-type]
        )

    assert not marker.exists()
    assert not hasattr(LocalExecutor, "_safe_exec")
