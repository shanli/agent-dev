"""适配器层：封装不同 API 协议的差异。

适配器职责只有四件事：
1. 把统一请求转换为协议特定的 payload
2. 解析 SSE/响应并归一成统一 chunk/result
3. 归一化 usage（input/output/cached/reasoning）
4. 把 HTTP 错误映射为 LLMError（阶段 + 错误码）

横切逻辑（重试、限流、观测）全部在 Gateway 层，适配器不实现。
"""
