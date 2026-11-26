"""transfer 工具治理测试：参数校验 / 业务预检 / 权限模式 / 审批转账 / 超时处理。

所有调用均通过 ToolRuntime.invoke()，不直接调用 transfer_handler。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

# 允许从 tests/ 子目录直接导入被测模块。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tool_governance_demo as mod  # noqa: E402
from tool_governance_demo import (  # noqa: E402
    ExecutionContext,
    PermissionMode,
    ToolCall,
    TransferArgs,
    build_governance,
)

# 转账工具在参数契约里使用 from_account / to_account 作为输入键。
VALID_ARGS = {
    "from_account": "ACC-A-123456",  # 余额 100000.0
    "to_account": "ACC-A-654321",    # 余额 5000.0
    "amount": 1000.0,
}
FROM_BALANCE = 100000.0
TO_BALANCE = 5000.0
AMOUNT = 1000.0


@pytest.fixture(autouse=True)
def _reset_accounts():
    """每个测试前后快照并恢复全局 ACCOUNTS，避免测试间相互污染。"""
    snapshot = dict(mod.ACCOUNTS)
    yield
    mod.ACCOUNTS.clear()
    mod.ACCOUNTS.update(snapshot)


def _ctx(**overrides) -> ExecutionContext:
    base = dict(
        trace_id="trace",
        user_id="tester",
        tenant_id="tenant_a",
        permissions=frozenset({"transfer:create"}),
        allowed_tools=frozenset({"create_transfer"}),
        mode=PermissionMode.DEFAULT,
    )
    base.update(overrides)
    return ExecutionContext(**base)


def _call(call_id: str, arguments: dict) -> ToolCall:
    return ToolCall(call_id, "create_transfer", arguments)


def _invoke(runtime, call: ToolCall, ctx: ExecutionContext):
    """同步包装：测试内用 asyncio.run 驱动异步 invoke。"""
    return asyncio.run(runtime.invoke(call, ctx))


# --------------------------------------------------------------------------- #
# 测试 1：参数校验
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "arguments",
    [
        # 注入未声明字段 approved / user_id（extra="forbid" 拦截）
        {**VALID_ARGS, "approved": True},
        {**VALID_ARGS, "user_id": "attacker"},
        # 账号格式非法（小写、缺位数）
        {"from_account": "acc-a-123456", "to_account": "ACC-A-654321", "amount": 100.0},
        {"from_account": "ACC-A-12345", "to_account": "ACC-A-654321", "amount": 100.0},
        # 金额为 0
        {"from_account": "ACC-A-123456", "to_account": "ACC-A-654321", "amount": 0},
        # 金额超过 10 万
        {"from_account": "ACC-A-123456", "to_account": "ACC-A-654321", "amount": 100001},
    ],
)
def test_transfer_invalid_argument(arguments):
    runtime, _approvals, _audit = build_governance()
    before = dict(mod.ACCOUNTS)

    result = _invoke(runtime, _call("c1", arguments), _ctx())

    assert result.ok is False
    assert result.error_code == "INVALID_ARGUMENT"
    assert mod.ACCOUNTS == before  # 一律不发生扣款


# --------------------------------------------------------------------------- #
# 测试 2：业务预检
# --------------------------------------------------------------------------- #
def test_transfer_precheck():
    runtime, _approvals, _audit = build_governance()
    before = dict(mod.ACCOUNTS)

    # 命中教学设定的限额区间 (50000, 80000] -> EXCEED_LIMIT
    r_limit = _invoke(
        runtime,
        _call("c1", {"from_account": "ACC-A-123456", "to_account": "ACC-A-654321", "amount": 60000.0}),
        _ctx(),
    )
    assert r_limit.error_code == "EXCEED_LIMIT"

    # 账户余额不足（金额不在限额区间内，进入余额校验）-> INSUFFICIENT_BALANCE
    r_balance = _invoke(
        runtime,
        _call("c2", {"from_account": "ACC-A-654321", "to_account": "ACC-A-123456", "amount": 10000.0}),
        _ctx(),
    )
    assert r_balance.error_code == "INSUFFICIENT_BALANCE"

    assert mod.ACCOUNTS == before  # 双方余额均不变


# --------------------------------------------------------------------------- #
# 测试 3：权限与模式限制
# --------------------------------------------------------------------------- #
def test_transfer_permission_and_mode():
    runtime, approvals, _audit = build_governance()
    before = dict(mod.ACCOUNTS)

    # 无转账权限（在工具白名单内，但缺 RBAC 权限）
    r_no_perm = _invoke(runtime, _call("c1", VALID_ARGS), _ctx(permissions=frozenset()))
    assert r_no_perm.error_code == "PERMISSION_DENIED"

    # 不在工具白名单
    r_not_whitelist = _invoke(runtime, _call("c2", VALID_ARGS), _ctx(allowed_tools=frozenset()))
    assert r_not_whitelist.error_code == "TOOL_NOT_ALLOWED"

    # Plan 模式：即使已获审批也不能绕过（plan 检查先于审批核销）
    approvals.approve(
        "ap_plan",
        user_id="tester",
        tenant_id="tenant_a",
        tool_name="create_transfer",
        args=TransferArgs(**VALID_ARGS),
    )
    r_plan = _invoke(
        runtime,
        _call("c3", VALID_ARGS),
        _ctx(mode=PermissionMode.PLAN, approval_id="ap_plan"),
    )
    assert r_plan.error_code == "PLAN_MODE_DENIED"

    assert mod.ACCOUNTS == before  # 均拒绝执行，不扣款


# --------------------------------------------------------------------------- #
# 测试 4：审批与正常转账
# --------------------------------------------------------------------------- #
def test_transfer_approval_and_success():
    runtime, approvals, audit = build_governance()
    args_changed = {**VALID_ARGS, "amount": 2000.0}

    # 1. 未审批即调用 -> CONFIRM（APPROVAL_REQUIRED）
    r_unapproved = _invoke(runtime, _call("c0", VALID_ARGS), _ctx())
    assert r_unapproved.error_code == "APPROVAL_REQUIRED"
    assert mod.ACCOUNTS[("tenant_a", "ACC-A-123456")] == FROM_BALANCE

    # 2. 签发审批，但调用时改金额 -> 原审批失效（digest 不匹配，审批未核销）
    approvals.approve(
        "ap1",
        user_id="tester",
        tenant_id="tenant_a",
        tool_name="create_transfer",
        args=TransferArgs(**VALID_ARGS),
    )
    r_changed = _invoke(runtime, _call("c1", args_changed), _ctx(approval_id="ap1"))
    assert r_changed.error_code == "APPROVAL_REQUIRED"
    assert mod.ACCOUNTS[("tenant_a", "ACC-A-123456")] == FROM_BALANCE  # 未扣款

    # 3. 用原参数转账 -> 成功；校验双方余额、账号脱敏与审计记录
    r_success = _invoke(runtime, _call("c2", VALID_ARGS), _ctx(approval_id="ap1"))
    assert r_success.ok is True
    assert r_success.error_code is None
    assert mod.ACCOUNTS[("tenant_a", "ACC-A-123456")] == FROM_BALANCE - AMOUNT
    assert mod.ACCOUNTS[("tenant_a", "ACC-A-654321")] == TO_BALANCE + AMOUNT
    # 账号脱敏
    assert r_success.content["from"] == "ACC-A-****3456"
    assert r_success.content["to"] == "ACC-A-****4321"
    # 审计记录：存在 create_transfer 成功记录
    success_records = [
        rec for rec in audit.records
        if rec.get("tool_name") == "create_transfer" and rec.get("ok") is True
    ]
    assert success_records, "应写入 create_transfer 成功审计记录"
    assert success_records[0]["content"]["from"] == "ACC-A-****3456"

    # 4. 同一审批不可重复使用
    r_reuse = _invoke(runtime, _call("c3", VALID_ARGS), _ctx(approval_id="ap1"))
    assert r_reuse.error_code == "APPROVAL_REQUIRED"
    assert mod.ACCOUNTS[("tenant_a", "ACC-A-123456")] == FROM_BALANCE - AMOUNT  # 未再次扣款


# --------------------------------------------------------------------------- #
# 测试 5：超时处理
# --------------------------------------------------------------------------- #
def test_transfer_timeout():
    runtime, approvals, audit = build_governance()
    # amount > 80000 触发 transfer_handler 内的 sleep(3.0)，超过策略 timeout_seconds=2
    args = {**VALID_ARGS, "amount": 90000.0}
    approvals.approve(
        "ap_to",
        user_id="tester",
        tenant_id="tenant_a",
        tool_name="create_transfer",
        args=TransferArgs(**args),
    )
    before = dict(mod.ACCOUNTS)

    result = _invoke(runtime, _call("c1", args), _ctx(approval_id="ap_to"))

    assert result.ok is False
    assert result.error_code == "TIMEOUT_UNKNOWN"
    assert mod.ACCOUNTS == before  # 超时点位于扣款之前，双方余额保持不变
    timeout_records = [rec for rec in audit.records if rec.get("error_code") == "TIMEOUT_UNKNOWN"]
    assert timeout_records, "应写入超时审计记录"


# --------------------------------------------------------------------------- #
# 离线演示：打印审计日志，展示账号脱敏与审批前 CONFIRM 状态
# --------------------------------------------------------------------------- #
async def run_offline_demo():
    runtime, approvals, audit = build_governance()

    def ctx(**overrides) -> ExecutionContext:
        base = dict(
            trace_id="demo",
            user_id="demo_user",
            tenant_id="tenant_a",
            permissions=frozenset({"transfer:create"}),
            allowed_tools=frozenset({"create_transfer"}),
        )
        base.update(overrides)
        return ExecutionContext(**base)

    print("=== 1. 未审批直接调用（应返回 CONFIRM / APPROVAL_REQUIRED）===")
    r0 = await runtime.invoke(ToolCall("d0", "create_transfer", VALID_ARGS), ctx())
    print(json.dumps({"ok": r0.ok, "error_code": r0.error_code, "content": r0.content},
                     ensure_ascii=False, indent=2))

    print("\n=== 2. 签发审批后用原参数转账（应成功，账号已脱敏）===")
    approvals.approve(
        "ap_demo",
        user_id="demo_user",
        tenant_id="tenant_a",
        tool_name="create_transfer",
        args=TransferArgs(**VALID_ARGS),
    )
    r1 = await runtime.invoke(ToolCall("d1", "create_transfer", VALID_ARGS), ctx(approval_id="ap_demo"))
    print(json.dumps({"ok": r1.ok, "error_code": r1.error_code, "content": r1.content},
                     ensure_ascii=False, indent=2))

    print("\n=== 3. 审计日志（应看到账号已脱敏 ACC-A-****3456）===")
    for rec in audit.records:
        print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run_offline_demo())
