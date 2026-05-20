from datetime import datetime

from src.schema.message import Message, Role
from src.util.lock import AsyncReadWriteLock


class Session:
    id: str
    work_dir: str
    created_at: datetime
    updated_at: datetime
    _history: list[Message]
    _lock: AsyncReadWriteLock

    def __init__(
            self,
            id: str,
            work_dir: str,
    ):
        self.id = id
        self.work_dir = work_dir
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self._history = []
        self._lock = AsyncReadWriteLock()

    async def append(self, messages: Message | list[Message]) -> None:
        if not messages:
            return

        await self._lock.acquire_write()
        try:
            if isinstance(messages, Message):
                messages = [messages]
            self._history.extend(messages)
            self.updated_at = datetime.now()

            # 持久化消息
        finally:
            await self._lock.release_write()

    async def get_working_memory(self, limit: int) -> list[Message]:
        await self._lock.acquire_read()
        try:
            total = len(self._history)
            if total <= limit or limit <= 0:
                return self._history[:]

            j = total - limit
            for i in range(total - limit, total):
                message = self._history[i]
                # 判断这是一条 tool result 消息
                if message.role == Role.ROLE_TOOL:
                    j = i + 1
                else:
                    break

            return self._history[j:]
        finally:
            await self._lock.release_read()

    async def is_inited(self) -> bool:
        await self._lock.acquire_read()
        try:
            return len(self._history) > 0
        finally:
            await self._lock.release_read()


class SessionManager:
    _sessions: dict[str, Session]
    _lock: AsyncReadWriteLock

    def __init__(self):
        self._sessions = {}
        self._lock = AsyncReadWriteLock()

    async def get_or_create(self, session_id: str, work_dir: str) -> Session:
        # 尝试使用读锁
        await self._lock.acquire_read()
        try:
            session = self._sessions.get(session_id, None)
            if session:
                return session
        finally:
            await self._lock.release_read()

        # 退回到写锁
        await self._lock.acquire_read()
        try:
            session = self._sessions.get(session_id, None)
            if session:
                return session

            session = Session(session_id, work_dir)
            self._sessions[session_id] = session
            return session
        finally:
            await self._lock.release_read()
