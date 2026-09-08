import asyncio
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_sandbox import PyodideSandbox

from workflow.engine.nodes.code.code_node import CodeNode
from workflow.engine.nodes.code.executor.langchain.langchain_executor import (
    LangchainExecutor,
    _FilePyodideSandbox,
)
from workflow.exception.e import CustomException
from workflow.exception.errors.err_code import CodeEnum


@pytest.fixture(autouse=True)
def skip_deno_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the SDK command builder without requiring installed Deno."""
    original_init = PyodideSandbox.__init__

    def initialize(sandbox: PyodideSandbox, **kwargs: Any) -> None:
        original_init(sandbox, skip_deno_check=True, **kwargs)

    monkeypatch.setattr(PyodideSandbox, "__init__", initialize)


def read_source_at_transport(
    sandbox: PyodideSandbox, code: str, memory_limit_mb: int | None = None
) -> tuple[Path, str]:
    command = sandbox._build_command(code, memory_limit_mb=memory_limit_mb)
    assert "-c" not in command
    assert code not in command
    source_path = Path(command[command.index("-f") + 1])
    source_bytes = source_path.read_bytes()
    assert source_bytes == code.encode("utf-8")
    assert f"--allow-read=node_modules,{source_path}" in command
    return source_path, source_bytes.decode("utf-8")


def test_file_command_preserves_sdk_permissions_limits_and_session(
    tmp_path: Path,
) -> None:
    source = "value = '张三\\n'\r\nprint(value)\r\n"
    source_path = tmp_path / "source.py"
    source_path.write_bytes(source.encode("utf-8"))
    sandbox = _FilePyodideSandbox(
        str(source_path),
        stateful=True,
        allow_env=False,
        allow_read=["node_modules", str(source_path)],
        allow_write=False,
        allow_net=False,
        allow_run=False,
        allow_ffi=False,
    )

    command = sandbox._build_command(
        source,
        session_bytes=b"session",
        session_metadata={"version": 1},
        memory_limit_mb=256,
    )

    assert command[:2] == ["deno", "run"]
    assert [arg for arg in command if arg.startswith("--allow-")] == [
        f"--allow-read=node_modules,{source_path}",
        "--allow-write=node_modules",
    ]
    assert "--node-modules-dir=auto" in command
    assert "--v8-flags=--max-old-space-size=256" in command
    assert "-s" in command
    assert json.loads(command[command.index("-b") + 1]) == list(b"session")
    assert json.loads(command[command.index("-m") + 1]) == {"version": 1}
    assert read_source_at_transport(sandbox, source)[1] == source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        '{\n  "name": "张三"\n}',
        r'{"name": "张三", "text": "first\nsecond"}',
        "first\r\nsecond\r\n",
        "张三 says: \"it's fine\"\n'''quoted'''\\folder\\new",
        {"nested": ["first\nsecond", r"first\nsecond", None, True]},
    ],
    ids=["multiline-json", "literal-backslash-n", "crlf", "quotes", "nested"],
)
async def test_code_node_preserves_parameter_and_source_escapes(
    monkeypatch: pytest.MonkeyPatch, data: Any
) -> None:
    source_paths: list[Path] = []

    async def execute_fixture(
        sandbox: PyodideSandbox, code: str, **kwargs: Any
    ) -> SimpleNamespace:
        path, source = read_source_at_transport(
            sandbox, code, kwargs["memory_limit_mb"]
        )
        source_paths.append(path)
        # Only this test's fixed main function is executed by the host Python.
        # Production execution remains entirely in the SDK's Deno subprocess.
        output = io.StringIO()
        with redirect_stdout(output):
            exec(compile(source, "<transport-fixture>", "exec"), {})
        return SimpleNamespace(
            status="success",
            stdout=output.getvalue(),
            stderr="",
        )

    monkeypatch.setenv("CODE_EXEC_TYPE", "langchain")
    monkeypatch.setattr(PyodideSandbox, "execute", execute_fixture)
    node = CodeNode(
        codeLanguage="python",
        input_identifier=["data"],
        output_identifier=["result", "newline", "literal", "matches"],
        code=r"""import re
def main(data):
    return {
        "result": data,
        "newline": "\n",
        "literal": r"\n",
        "matches": re.findall(r"\n", "first\nsecond"),
    }
""",
        appId="app-1",
        uid="user-1",
        node_id="ifly-code::source-transport",
    )
    span = MagicMock()
    span.add_info_event_async = AsyncMock()

    result = await node.execute_code({"data": data}, span)

    assert result == {
        "result": data,
        "newline": "\n",
        "literal": r"\n",
        "matches": ["\n"],
    }
    assert len(source_paths) == 1
    assert not source_paths[0].exists()
    assert not source_paths[0].parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["sdk-error", "timeout", "cancelled"])
async def test_source_file_is_removed_when_execution_fails(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source_paths: list[Path] = []

    async def fail_execution(
        sandbox: PyodideSandbox, code: str, **kwargs: Any
    ) -> SimpleNamespace:
        path, _ = read_source_at_transport(sandbox, code)
        source_paths.append(path)
        if failure == "sdk-error":
            raise RuntimeError("Deno process failed")
        if failure == "cancelled":
            raise asyncio.CancelledError()
        return SimpleNamespace(
            status="error",
            stdout="",
            stderr="Execution timed out after 10 seconds",
        )

    monkeypatch.setattr(PyodideSandbox, "execute", fail_execution)
    expected_error = (
        asyncio.CancelledError if failure == "cancelled" else CustomException
    )

    with pytest.raises(expected_error) as error:
        await LangchainExecutor().execute(
            "python",
            "print('ok')",
            10,
            MagicMock(),
        )

    if failure != "cancelled":
        expected_code = (
            CodeEnum.CODE_EXECUTION_TIMEOUT_ERROR
            if failure == "timeout"
            else CodeEnum.CODE_EXECUTION_ERROR
        )
        assert error.value.code == expected_code.code
    assert len(source_paths) == 1
    assert not source_paths[0].exists()
    assert not source_paths[0].parent.exists()


@pytest.mark.asyncio
async def test_concurrent_executions_keep_source_files_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_paths: list[Path] = []
    both_started = asyncio.Event()

    async def overlapping_execution(
        sandbox: PyodideSandbox, code: str, **kwargs: Any
    ) -> SimpleNamespace:
        path, _ = read_source_at_transport(sandbox, code)
        source_paths.append(path)
        if len(source_paths) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        assert path.read_bytes() == code.encode("utf-8")
        return SimpleNamespace(status="success", stdout=code, stderr="")

    monkeypatch.setattr(PyodideSandbox, "execute", overlapping_execution)
    sources = ["print('first\\n')\r\n", "print('第二个\\n')\n"]

    results = await asyncio.gather(
        *[
            LangchainExecutor().execute("python", source, 10, MagicMock())
            for source in sources
        ]
    )

    assert results == sources
    assert len(set(source_paths)) == 2
    assert all(not path.exists() for path in source_paths)
    assert all(not path.parent.exists() for path in source_paths)
