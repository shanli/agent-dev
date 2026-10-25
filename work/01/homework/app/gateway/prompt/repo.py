"""提示词版本仓库：数据库 CRUD。"""

from __future__ import annotations

from typing import Optional

from datetime import datetime

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PromptVersion


class PromptRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_version(self, name: str, version: Optional[int]) -> Optional[PromptVersion]:
        """取指定版本；version=None 时取当前激活版本。"""
        if version is not None:
            stmt = select(PromptVersion).where(
                and_(PromptVersion.name == name, PromptVersion.version == version)
            )
        else:
            stmt = select(PromptVersion).where(
                and_(PromptVersion.name == name, PromptVersion.is_active.is_(True))
            )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_versions(self, name: str) -> list[PromptVersion]:
        stmt = (
            select(PromptVersion)
            .where(PromptVersion.name == name)
            .order_by(PromptVersion.version.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def create_version(
        self, name: str, template: str, variables: Optional[list[str]] = None
    ) -> PromptVersion:
        """创建新版本（版本号自增，不自动激活）。"""
        result = await self._session.execute(
            select(func.coalesce(func.max(PromptVersion.version), 0)).where(
                PromptVersion.name == name
            )
        )
        next_version = (result.scalar_one() or 0) + 1
        pv = PromptVersion(
            name=name,
            version=next_version,
            template=template,
            variables=variables,
            is_active=False,
        )
        self._session.add(pv)
        await self._session.commit()
        await self._session.refresh(pv)
        return pv

    async def activate(self, name: str, version: int) -> None:
        """激活指定版本（同 name 下其他版本置为非激活）。版本不可变，回滚 = 切换激活。"""
        await self._session.execute(
            PromptVersion.__table__.update()
            .where(PromptVersion.name == name)
            .values(is_active=False)
        )
        await self._session.execute(
            PromptVersion.__table__.update()
            .where(and_(PromptVersion.name == name, PromptVersion.version == version))
            .values(is_active=True)
        )
        await self._session.commit()
