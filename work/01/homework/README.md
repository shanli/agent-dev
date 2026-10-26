# LLM 统一模型调用服务

一个基于 FastAPI 的 LLM 网关服务，用统一接口屏蔽底层协议差异，将 **OpenAI Responses** 与 **Anthropic Messages** 两种协议的模型纳入同一套调用、重试、限流、结构化输出与可观测体系。

## 功能特性

- **双协议适配**：逻辑模型名 → 协议适配器路由，目前支持 OpenAI Responses、Anthropic Messages
- **候选池 Fallback**：本地限流 / 上游 429 时自动切换候选模型，保证可用性
- **阶段化错误处理**：错误按管线阶段分类，重试与 fallback 策略由错误码矩阵决定（内容拒答红线禁止换模型绕过）
- **指数退避重试**：最多 2 次重试（共 3 次尝试），尊重上游 `Retry-After`，截止时间预算不足时不再烧调用
- **双令牌桶限流**：按模型独立的 RPM + TPM 限流，超限返回 429 并附带 `retry_after_ms`
- **流式调用**：SSE 归一化；首 token 前可安全重试 + fallback，流中错误只发 error chunk 不静默重试
- **结构化输出**：JSON Schema 校验 + 独立修复预算（1 次），修复失败返回 502
- **Prompt 版本管理**：版本不可变，回滚 = 切换激活版本
- **九维可观测**：关联 / 路由 / Prompt / 用量 / 延迟 / 弹性 / 结果 / 错误 / 成本，落库 + Prometheus 指标

## 目录结构

```
app/
├── adapters/            # 协议适配器
│   ├── base.py          # 适配器抽象基类 + 统一数据类型
│   ├── openai_responses.py
│   ├── anthropic_messages.py
│   ├── registry.py      # 模型 → 适配器路由
│   └── sse.py           # SSE 解析
├── api/routes.py        # HTTP 路由
├── core/
│   ├── config.py        # 配置加载（pydantic-settings + models.yaml）
│   ├── errors.py        # 阶段化错误模型 + 重试/fallback 策略矩阵
│   └── types.py         # 统一请求/响应/流式 chunk 类型
├── db/                  # SQLAlchemy 模型 + 会话
├── gateway/
│   ├── gateway.py       # 编排门面（Prompt/限流/重试/路由/结构化/可观测）
│   ├── prompt/          # Prompt 版本解析 + 渲染 + 变量校验
│   ├── resilience/      # 重试执行器 + 双令牌桶限流
│   └── structured/      # JSON Schema 校验 + 修复
├── observability/       # 调用记录落库 + Prometheus 指标
└── main.py              # 应用入口
tests/                   # pytest 测试集
models.yaml              # 模型注册表
docker-compose.yml       # PostgreSQL
```

## 环境要求

- Python >= 3.9
- PostgreSQL 16（或用 SQLite 降级）

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
# PostgreSQL 驱动
pip install asyncpg
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

`.env` 关键字段：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 上游 API Key | 无 |
| `DATABASE_URL` | 数据库连接串 | `postgresql+asyncpg://llm:llm@localhost:5432/llm` |
| `MODELS_PATH` | 模型配置文件路径 | `./models.yaml` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

> 使用 SQLite 降级（无需 PostgreSQL）：`DATABASE_URL=sqlite+aiosqlite:///./llm_dev.db`

### 3. 启动数据库

```bash
docker compose up -d
```

### 4. 启动服务

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动后访问：

- 根路径 `/` → 自动跳转 Swagger 文档
- `/docs` — Swagger 交互文档
- `/health` — 健康检查
- `/metrics` — Prometheus 指标

## 模型配置（models.yaml）

```yaml
models:
  deepseek-v4-pro:
    provider: openai_responses          # 适配器类型
    actual_model: deepseek-v4-pro       # 发给上游的真实模型名
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
    rpm: 60                             # 每分钟请求数限流
    tpm: 100000                         # 每分钟 token 数限流
    price_per_million:                  # 每百万 token 单价（USD），用于成本估算
      input: 0.27
      output: 1.10
      cached: 0.07
    candidates:                         # 候选模型，fallback 时按顺序切换
      - deepseek-v4-flash
```

## API 接口

### 非流式对话

```http
POST /v1/chat
Content-Type: application/json

{
  "model": "deepseek-v4-pro",
  "messages": [{"role": "user", "content": "你好"}],
  "temperature": 0.7,
  "max_tokens": 1024,
  "response_schema": null,        // 传入 JSON Schema 即启用结构化输出
  "stream": false,
  "timeout_ms": 60000,
  "trace_id": "optional-trace-id"
}
```

响应：

```json
{
  "id": "resp-xxx",
  "logical_model": "deepseek-v4-pro",
  "actual_model": "deepseek-v4-pro",
  "content": "你好！有什么可以帮你的？",
  "usage": {"input_tokens": 10, "output_tokens": 8, "cached_tokens": 0, "reasoning_tokens": 0},
  "finish_reason": "stop",
  "total_latency_ms": 320,
  "ttft_ms": null,
  "attempt": 1,
  "provider_request_id": "req-xxx"
}
```

### 流式对话

```http
POST /v1/chat/stream
Content-Type: application/json

{ "model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "你好"}], "stream": true }
```

返回 `text/event-stream`，chunk 类型：

| type | 字段 | 说明 |
|---|---|---|
| `text_delta` | `delta` | 文本增量 |
| `usage` | `usage` | token 用量 |
| `done` | `finish_reason` | 流结束 |
| `error` | `error_code`, `partial_content` | 流中断，附带已流出的部分内容 |

### Prompt 版本管理

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/prompts/{name}/versions` | 创建新版本（不自动激活） |
| POST | `/v1/prompts/{name}/versions/{version}/activate` | 激活指定版本（回滚 = 切激活） |
| GET | `/v1/prompts/{name}/versions` | 列出所有版本 |

Prompt 模板用 `{{var}}` 语法引用变量，调用时通过 `prompt_ref` 引用：

```json
{
  "model": "deepseek-v4-pro",
  "prompt_ref": {
    "name": "summarize",
    "version": 1,
    "variables": {"text": "待总结内容"}
  }
}
```

## 错误码

| code | HTTP | 可重试 | 可 fallback | 说明 |
|---|---|---|---|---|
| `AUTH_FAILED` | 401 | ✗ | ✗ | 认证失败 |
| `INVALID_REQUEST` | 400 | ✗ | ✗ | 请求校验失败 |
| `PROMPT_VARIABLE_MISSING` | 422 | ✗ | ✗ | Prompt 变量缺失 |
| `NO_COMPATIBLE_MODEL` | 404 | ✗ | ✗ | 未注册的模型 |
| `RATE_LIMITED` | 429 | ✗ | ✓ | 本地限流，可转投候选池 |
| `UPSTREAM_RATE_LIMITED` | - | ✓ | ✓ | 上游 429 |
| `UPSTREAM_TIMEOUT` | 504 | ✓ | ✓ | 上游超时 |
| `UPSTREAM_OVERLOADED` | - | ✓ | ✓ | 上游过载（529） |
| `UPSTREAM_UNAVAILABLE` | - | ✓ | ✗ | 上游 5xx |
| `SCHEMA_VALIDATION_FAILED` | 502 | ✗ | ✗ | 结构化输出校验失败 |
| `CONTENT_FILTERED` | 422 | ✗ | ✗ | 内容拒答（红线，禁止换模型绕过） |
| `DEADLINE_EXCEEDED` | 504 | ✗ | ✗ | 截止时间预算耗尽 |

错误响应体：

```json
{"error": {"code": "RATE_LIMITED", "stage": "queue", "message": "...", "retry_after_ms": 1050}}
```

## 测试

```bash
pytest tests/
```

测试集覆盖：

- `test_adapters.py` — 适配器契约（请求翻译、响应归一、错误映射）
- `test_retry.py` — 重试次数、退避策略、截止时间、内容拒答红线
- `test_router.py` — 注册表路由、HTTP 错误码映射
- `test_streaming.py` — SSE 事件序列、WAITING 期错误处理
- `test_structured.py` — Schema 校验、修复流程
- `test_observability.py` — Trace 记录、成本台账、Prometheus 指标
- `test_ratelimit.py` — 双令牌桶耗尽、retry_after、未注册模型
- `test_fallback.py` — 候选池切换、超时映射

## 许可证

内部使用
