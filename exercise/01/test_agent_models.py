"""测试 agent_models：JSON 解析、序列化、字段对比、校验器、exclude 行为。"""

import json

from agent_models import (
    AgentListResponse,
    AgentResponse,
    AgentStatus,
    ChatRequest,
    ChatResponse,
    CreateAgentRequest,
    ErrorResponse,
    Message,
    MessageRole,
    ToolCall,
    ToolExecuteRequest,
    ToolExecuteResponse,
)


def test_create_agent_round_trip() -> None:
    """完整请求 JSON → model_validate_json → 序列化回 JSON → 对比字段。"""
    payload = {
        "name": "代码助手",
        "model": "deepseek-v4-pro",
        "system_prompt": "你是一个专业的代码审查助手",
        "temperature": 0.3,
        "max_tokens": 4096,
        "api_token": "sk-secret-12345",
    }
    raw_json = json.dumps(payload, ensure_ascii=False)

    # 1. 从 JSON 解析
    agent = CreateAgentRequest.model_validate_json(raw_json)
    assert agent.name == "代码助手"
    assert agent.temperature == 0.3
    assert agent.api_token == "sk-secret-12345"  # 内存中可访问

    # 2. 序列化回 JSON（api_token 因 exclude=True 被排除）
    dumped = agent.model_dump()
    assert "api_token" not in dumped, "敏感字段应被 exclude"
    assert dumped["name"] == "代码助手"

    # 3. model_dump_json 同样排除
    dumped_json = agent.model_dump_json()
    assert "api_token" not in dumped_json
    assert "代码助手" in dumped_json

    print("[OK] CreateAgentRequest round-trip，api_token 已被 exclude")


def test_chat_request_validators() -> None:
    """测试 field_validator（content 去空白）和 model_validator（首条消息角色）。"""
    # 正常：首条 user
    req = ChatRequest.model_validate({
        "agent_id": "a-001",
        "messages": [{"role": "user", "content": "  你好  "}],
    })
    assert req.messages[0].content == "你好"  # field_validator 去空白
    print("[OK] ChatRequest field_validator 去空白生效")

    # 校验失败：首条 assistant
    try:
        ChatRequest.model_validate({
            "agent_id": "a-001",
            "messages": [{"role": "assistant", "content": "hi"}],
        })
        assert False, "应抛出校验错误"
    except Exception as e:
        assert "首条消息" in str(e)
        print(f"[OK] ChatRequest model_validator 拦截首条非 user/system: {e}")

    # 校验失败：空内容被 strip 后为空
    try:
        ChatRequest.model_validate({
            "agent_id": "a-001",
            "messages": [{"role": "user", "content": "   "}],
        })
        assert False, "应抛出校验错误"
    except Exception as e:
        assert "不能为空" in str(e)
        print(f"[OK] Message field_validator 拦截空白内容: {e}")


def test_tool_call_message_rule() -> None:
    """测试 Message 的 model_validator：role=tool 必须有 tool_call_id。"""
    # tool 角色无 tool_call_id → 失败
    try:
        Message.model_validate({"role": "tool", "content": "result"})
        assert False
    except Exception as e:
        assert "tool_call_id" in str(e)
        print(f"[OK] Message model_validator 拦截 tool 缺 tool_call_id: {e}")

    # tool 角色带 tool_call_id → 成功
    msg = Message.model_validate({
        "role": "tool",
        "content": "result",
        "tool_call_id": "call_1",
    })
    assert msg.tool_call_id == "call_1"
    print("[OK] Message tool 角色带 tool_call_id 通过")


def test_extra_forbid() -> None:
    """请求模型 extra='forbid'：多余字段应报错。"""
    try:
        CreateAgentRequest.model_validate({
            "name": "x",
            "model": "m",
            "api_token": "t",
            "unknown_field": 123,
        })
        assert False
    except Exception as e:
        assert "unknown_field" in str(e)
        print(f"[OK] extra='forbid' 拦截多余字段: {e}")


def test_chat_response_nested() -> None:
    """测试嵌套模型：ChatResponse 内含 Message + ToolCall 列表。"""
    resp = ChatResponse.model_validate({
        "agent_id": "a-001",
        "message": {
            "role": "assistant",
            "content": "我来帮你查询",
            "tool_calls": [
                {"id": "c1", "name": "search", "arguments": {"q": "python"}},
            ],
        },
        "tool_calls": [
            {"id": "c1", "name": "search", "arguments": {"q": "python"}},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    })
    assert isinstance(resp.message.tool_calls[0], ToolCall)
    assert resp.tool_calls[0].name == "search"
    assert resp.usage["output_tokens"] == 20
    print("[OK] ChatResponse 嵌套模型（Message + ToolCall + usage）解析正确")


def test_agent_list_pagination() -> None:
    """测试 AgentListResponse 嵌套 AgentResponse 列表 + Pagination。"""
    resp = AgentListResponse.model_validate({
        "items": [
            {
                "id": "a-001",
                "name": "助手1",
                "model": "gpt-4",
                "system_prompt": "hi",
                "temperature": 0.7,
                "max_tokens": 2048,
                "status": "idle",
                "created_at": "2026-10-27T10:00:00",
            },
        ],
        "pagination": {"page": 1, "page_size": 20, "total": 1},
    })
    assert resp.items[0].status == AgentStatus.IDLE
    assert resp.pagination.total == 1
    print("[OK] AgentListResponse 嵌套 AgentResponse + Pagination 解析正确")


def test_full_json_round_trip() -> None:
    """端到端：构造完整请求 JSON → validate_json → dump_json → 字段一致性对比。"""
    payload = {
        "agent_id": "agent-999",
        "messages": [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "查询天气"},
            {
                "role": "assistant",
                "content": "我来调用工具",
                "tool_calls": [
                    {"id": "tc-1", "name": "get_weather", "arguments": {"city": "北京"}},
                ],
            },
            {"role": "tool", "content": "晴 25°C", "tool_call_id": "tc-1"},
        ],
        "stream": True,
    }
    raw_json = json.dumps(payload, ensure_ascii=False)

    req = ChatRequest.model_validate_json(raw_json)
    out = json.loads(req.model_dump_json())

    # 对比关键字段
    assert out["agent_id"] == payload["agent_id"]
    assert out["stream"] is True
    assert len(out["messages"]) == 4
    assert out["messages"][0]["role"] == "system"
    assert out["messages"][3]["tool_call_id"] == "tc-1"
    assert out["messages"][2]["tool_calls"][0]["name"] == "get_weather"

    print("[OK] ChatRequest 端到端 JSON round-trip 字段一致")


def test_error_response() -> None:
    """测试 ErrorResponse。"""
    err = ErrorResponse.model_validate({
        "code": "VALIDATION_ERROR",
        "message": "请求参数校验失败",
        "details": [{"field": "temperature", "message": "必须在 0-2 之间"}],
        "request_id": "req-abc",
    })
    assert err.code == "VALIDATION_ERROR"
    assert err.details[0].field == "temperature"
    print("[OK] ErrorResponse 解析正确")


if __name__ == "__main__":
    test_create_agent_round_trip()
    test_chat_request_validators()
    test_tool_call_message_rule()
    test_extra_forbid()
    test_chat_response_nested()
    test_agent_list_pagination()
    test_full_json_round_trip()
    test_error_response()
    print("\n全部测试通过 ✅")
