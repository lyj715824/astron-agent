import json
from typing import Any

import pytest

from workflow.extensions.otlp.log_trace import workflow_log as workflow_log_module
from workflow.extensions.otlp.log_trace.node_log import NodeLog
from workflow.extensions.otlp.log_trace.workflow_log import WorkflowLog


class RecordingStorage:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes, str | None]] = []

    def upload_file(
        self, filename: str, file_bytes: bytes, bucket_name: str | None = None
    ) -> str:
        self.uploads.append((filename, file_bytes, bucket_name))
        return f"https://trace.invalid/{len(self.uploads)}.txt"


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> RecordingStorage:
    service = RecordingStorage()
    monkeypatch.setattr(workflow_log_module, "get_oss_service", lambda: service)
    monkeypatch.setenv("OSS_BUCKET_NAME", "trace-test")
    return service


def decode_variables(data: dict[str, Any], field: str) -> dict[str, Any]:
    assert isinstance(data[field], list)
    assert all(isinstance(variable, str) for variable in data[field])
    return {
        variable["name"]: variable["value"]
        for variable in (json.loads(raw) for raw in data[field])
    }


def test_deep_variables_externalize_values_without_losing_wrappers(
    storage: RecordingStorage,
) -> None:
    content = "中文输出" * 2048
    node = NodeLog(sid="sid", id="node-log", node_id="spark-llm::1")
    node.append_input_data("history", content)
    node.append_output_data("answer", content)
    node.append_output_data("short_answer", "正常")
    workflow = WorkflowLog(sid="sid", flow_id="flow", trace=[node])

    payload = json.loads(workflow.to_json())
    data = payload["trace"][0]["data"]

    assert decode_variables(data, "input_vars") == {
        "history": "https://trace.invalid/1.txt"
    }
    assert decode_variables(data, "output_vars") == {
        "answer": "https://trace.invalid/1.txt",
        "short_answer": "正常",
    }
    assert data["input"] == {}
    assert data["output"] == {}
    assert len(storage.uploads) == 1
    assert storage.uploads[0][1:] == (content.encode("utf-8"), "trace-test")


def test_nested_messages_keep_their_structure_after_one_json_decode(
    storage: RecordingStorage,
) -> None:
    content = "历史会话" * 2048
    messages = [
        {"role": "system", "content": "system prompt"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": {"url": "https://image.invalid/a"}},
            ],
        },
    ]
    node = NodeLog(sid="sid")
    node.append_config_data({"message": messages})
    workflow = WorkflowLog(sid="sid", flow_id="flow", trace=[node])

    payload = json.loads(workflow.to_json())
    encoded_messages = payload["trace"][0]["data"]["config"]["message"]

    assert isinstance(encoded_messages, str)
    assert json.loads(encoded_messages) == [
        {"role": "system", "content": "system prompt"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "https://trace.invalid/1.txt"},
                {"type": "image_url", "image_url": {"url": "https://image.invalid/a"}},
            ],
        },
    ]
    assert len(storage.uploads) == 1


@pytest.mark.parametrize(
    ("content", "should_upload"),
    [
        ("a" * 5120, False),
        ("a" * 5121, True),
        ("中" * 1706 + "ab", False),
        ("中" * 1707, True),
    ],
    ids=["ascii-at-limit", "ascii-over-limit", "utf8-at-limit", "utf8-over-limit"],
)
def test_externalization_limit_counts_utf8_bytes(
    storage: RecordingStorage, content: str, should_upload: bool
) -> None:
    node = NodeLog(sid="sid")
    node.append_output_data("answer", content)
    workflow = WorkflowLog(sid="sid", flow_id="flow", trace=[node])

    payload = json.loads(workflow.to_json())
    value = decode_variables(payload["trace"][0]["data"], "output_vars")["answer"]

    assert value == ("https://trace.invalid/1.txt" if should_upload else content)
    assert len(storage.uploads) == int(should_upload)


def test_duplicate_content_uploads_once_per_serialization_and_preserves_model(
    storage: RecordingStorage,
) -> None:
    content = "重复内容" * 2048
    node = NodeLog(sid="sid", llm_output=content)
    node.append_input_data("history", content)
    node.append_output_data("answer", content)
    node.append_config_data({"message": [{"role": "assistant", "content": content}]})
    workflow = WorkflowLog(sid="sid", flow_id="flow", answer=content, trace=[node])
    before = workflow.model_dump(mode="json")

    first = json.loads(workflow.to_json())

    assert len(storage.uploads) == 1
    assert first["answer"] == first["trace"][0]["llm_output"]
    assert (
        decode_variables(first["trace"][0]["data"], "output_vars")["answer"]
        == first["answer"]
    )
    assert workflow.model_dump(mode="json") == before

    second = json.loads(workflow.to_json())

    assert len(storage.uploads) == 2
    assert second["answer"] == "https://trace.invalid/2.txt"
    assert workflow.model_dump(mode="json") == before


def test_small_trace_keeps_console_compatible_fields(
    storage: RecordingStorage,
) -> None:
    node = NodeLog(
        sid="sid",
        id="log-1",
        node_id="spark-llm::1",
        node_name="问答",
        node_type="spark-llm",
        next_log_ids={"log-2"},
        start_time=1000,
        end_time=1020,
        duration=20,
    )
    node.append_input_data("query", "你好")
    node.append_output_data("answer", "您好")
    node.append_config_data({"message": [{"role": "user", "content": "你好"}]})
    node.append_usage_data(
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    )
    workflow = WorkflowLog(sid="sid", flow_id="flow", trace=[node])

    payload = json.loads(workflow.to_json())
    result = payload["trace"][0]

    assert (payload["sid"], payload["flow_id"], payload["sub"]) == (
        "sid",
        "flow",
        "workflow",
    )
    assert result["id"] == "log-1"
    assert result["node_id"] == "spark-llm::1"
    assert result["node_name"] == "问答"
    assert result["next_log_ids"] == ["log-2"]
    assert (result["start_time"], result["end_time"], result["duration"]) == (
        1000,
        1020,
        20,
    )
    assert result["running_status"] is True
    assert result["data"]["input_vars"] == ['{"name": "query", "value": "你好"}']
    assert result["data"]["output_vars"] == ['{"name": "answer", "value": "您好"}']
    assert json.loads(result["data"]["config"]["message"]) == [
        {"role": "user", "content": "你好"}
    ]
    assert int(result["data"]["usage"]["total_tokens"]) == 5
    assert storage.uploads == []


def test_multiturn_long_answers_keep_nodes_and_usage_in_small_payloads(
    storage: RecordingStorage,
) -> None:
    history: list[dict[str, str]] = []
    for turn in range(3):
        # Deliberately exceed one MiB to exercise the formerly bypassed deep fields.
        answer = f"第{turn}轮建议：" + "健康建议" * 100_000
        node_ids = [
            "node-start::1",
            "decision-making::1",
            "spark-llm::1",
            "node-end::1",
        ]
        nodes = [
            NodeLog(sid=f"sid-{turn}", id=f"log-{turn}-{index}", node_id=node_id)
            for index, node_id in enumerate(node_ids)
        ]
        for node, following in zip(nodes, nodes[1:]):
            node.set_next_node_id(following.id)
        model = nodes[2]
        model.append_input_data("query", f"问题{turn}")
        model.append_input_data("chatHistory", history)
        model.append_output_data("answer", answer)
        model.llm_output = answer
        model.append_config_data(
            {"message": history + [{"role": "user", "content": f"问题{turn}"}]}
        )
        model.append_usage_data(
            {"prompt_tokens": 800, "completion_tokens": 2800, "total_tokens": 3600}
        )
        nodes[3].append_input_data("answer", answer)
        nodes[3].append_output_data("answer", answer)
        workflow = WorkflowLog(
            sid=f"sid-{turn}",
            flow_id="flow",
            chat_id="chat",
            answer=answer,
            trace=nodes,
        )
        workflow.set_end()

        serialized = workflow.to_json()
        payload = json.loads(serialized)

        assert len(serialized.encode("utf-8")) < 1024 * 1024
        assert payload["chat_id"] == "chat"
        assert payload["sid"] == f"sid-{turn}"
        assert [node["node_id"] for node in payload["trace"]] == node_ids
        assert [node["id"] for node in payload["trace"]] == [node.id for node in nodes]
        assert payload["trace"][2]["next_log_ids"] == [nodes[3].id]
        assert payload["usage"]["total_tokens"] == 3600
        assert int(payload["trace"][2]["data"]["usage"]["completion_tokens"]) == 2800
        model_output = decode_variables(payload["trace"][2]["data"], "output_vars")
        end_input = decode_variables(payload["trace"][3]["data"], "input_vars")
        assert model_output["answer"] == end_input["answer"] == payload["answer"]
        history.extend(
            [
                {"role": "user", "content": f"问题{turn}"},
                {"role": "assistant", "content": answer},
            ]
        )
