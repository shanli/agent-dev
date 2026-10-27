"""模拟数据库：用内存 dict 存储，并持久化到 JSON 文件。

正式课替换为 SQLAlchemy/PostgreSQL。
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_api.models import AgentResponse, AgentStatus

# 数据持久化文件路径（与 db.py 同目录）
_DATA_FILE = Path(__file__).resolve().parent / "agents_data.json"

_logger = logging.getLogger(__name__)


class InMemoryDB:
    """带 JSON 持久化的内存存储，仅用于教学演示。"""

    def __init__(self) -> None:
        self._agents: dict[str, AgentResponse] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._load()

    # ── 持久化 ──

    def _load(self) -> None:
        """服务启动时从 JSON 文件加载数据；文件缺失或损坏时以空库运行。"""
        if not _DATA_FILE.exists():
            return
        try:
            raw = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
            self._counter = int(raw.get("counter", 0))
            for agent_id, data in raw.get("agents", {}).items():
                self._agents[agent_id] = AgentResponse(**data)
            _logger.info("已从 %s 加载 %d 条 Agent 数据", _DATA_FILE.name, len(self._agents))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            _logger.warning("数据文件加载失败，以空库启动: %s", e)
            self._agents = {}
            self._counter = 0

    def _save(self) -> None:
        """每次修改后将全量数据写入 JSON 文件。"""
        payload = {
            "counter": self._counter,
            "agents": {
                aid: agent.model_dump(mode="json") for aid, agent in self._agents.items()
            },
        }
        _DATA_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _next_id(self) -> str:
        self._counter += 1
        return f"agent-{self._counter:04d}"

    # ── CRUD ──

    def create(self, data: dict) -> AgentResponse:
        with self._lock:
            agent_id = self._next_id()
            agent = AgentResponse(
                id=agent_id,
                name=data["name"],
                model=data["model"],
                system_prompt=data.get("system_prompt", "你是一个有用的 AI 助手"),
                temperature=data.get("temperature", 0.7),
                max_tokens=data.get("max_tokens", 2048),
                status=AgentStatus.IDLE,
                created_at=datetime.now(timezone.utc),
            )
            self._agents[agent_id] = agent
            self._save()
            return agent

    def get(self, agent_id: str) -> Optional[AgentResponse]:
        return self._agents.get(agent_id)

    def list_all(self) -> list[AgentResponse]:
        return list(self._agents.values())

    def search(self, query: str) -> list[AgentResponse]:
        """按名称模糊搜索（大小写不敏感的子串匹配）。"""
        q = query.strip().lower()
        if not q:
            return []
        return [a for a in self._agents.values() if q in a.name.lower()]

    def update(self, agent_id: str, data: dict) -> Optional[AgentResponse]:
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return None
            updated = agent.model_copy(update=data)
            self._agents[agent_id] = updated
            self._save()
            return updated

    def delete(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                self._save()
                return True
            return False


# 全局单例
db = InMemoryDB()
