import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from workflow.extensions.otlp.log_trace.node_log import NodeLog
from workflow.extensions.otlp.log_trace.workflow_log import WorkflowLog
from workflow.extensions.otlp.trace.span import Span
from workflow.service import ops_service


class RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: Any) -> None:
        self.messages.append(message.format(*args))

    def error(self, message: str, *args: Any) -> None:
        self.messages.append(message.format(*args))


@pytest.fixture
def report_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Mock, RecordingLogger]:
    class ImmediateThread:
        def __init__(self, *, target: Any, daemon: bool) -> None:
            self.target = target
            assert daemon is True

        def start(self) -> None:
            self.target()

    producer = Mock()
    logger = RecordingLogger()
    # Patch the module binding rather than the shared threading.Thread object.
    monkeypatch.setattr(
        ops_service, "threading", SimpleNamespace(Thread=ImmediateThread)
    )
    monkeypatch.setattr(ops_service, "get_kafka_producer_service", lambda: producer)
    monkeypatch.setattr(ops_service, "logger", logger)
    monkeypatch.setenv("KAFKA_TOPIC", "trace-test")
    return producer, logger


def make_workflow() -> WorkflowLog:
    node = NodeLog(sid="sid-test", id="log-test", node_id="node-end::1")
    node.append_output_data("answer", "PRIVATE_用户会话正文")
    return WorkflowLog(
        sid="sid-test", flow_id="flow-test", answer="PRIVATE_用户会话正文", trace=[node]
    )


def test_report_sends_once_and_logs_only_payload_metadata(
    report_dependencies: tuple[Mock, RecordingLogger],
) -> None:
    producer, logger = report_dependencies
    workflow = make_workflow()

    ops_service.kafka_report(workflow, Mock(spec=Span))

    producer.send.assert_called_once()
    topic, serialized = producer.send.call_args.args
    assert topic == "trace-test"
    payload = json.loads(serialized)
    assert payload["answer"] == "PRIVATE_用户会话正文"
    assert payload["status"] == {"code": 0, "message": "success"}
    assert payload["trace"][0]["id"] == "log-test"
    text = "\n".join(logger.messages)
    assert "sid=sid-test" in text
    assert "flow_id=flow-test" in text
    assert f"payload_bytes={len(serialized.encode('utf-8'))}" in text
    assert "nodes=1" in text
    assert "PRIVATE_用户会话正文" not in text


def test_send_failure_reports_stage_and_bytes_without_retry_or_payload(
    report_dependencies: tuple[Mock, RecordingLogger],
) -> None:
    producer, logger = report_dependencies
    producer.send.side_effect = RuntimeError("MSG_SIZE_TOO_LARGE")

    ops_service.kafka_report(make_workflow(), Mock(spec=Span))

    producer.send.assert_called_once()
    serialized = producer.send.call_args.args[1]
    error = logger.messages[-1]
    assert "sid=sid-test" in error
    assert "flow_id=flow-test" in error
    assert "stage=kafka_send" in error
    assert f"payload_bytes={len(serialized.encode('utf-8'))}" in error
    assert "MSG_SIZE_TOO_LARGE" in error
    assert all("PRIVATE_用户会话正文" not in message for message in logger.messages)


def test_serialization_failure_does_not_send_and_identifies_stage(
    monkeypatch: pytest.MonkeyPatch,
    report_dependencies: tuple[Mock, RecordingLogger],
) -> None:
    producer, logger = report_dependencies

    def fail_serialization(self: WorkflowLog) -> str:
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(WorkflowLog, "to_json", fail_serialization)

    ops_service.kafka_report(make_workflow(), Mock(spec=Span))

    producer.send.assert_not_called()
    error = logger.messages[-1]
    assert "sid=sid-test" in error
    assert "flow_id=flow-test" in error
    assert "stage=serialize" in error
    assert "payload_bytes=0" in error
    assert "storage unavailable" in error
    assert "PRIVATE_用户会话正文" not in error
