from typing_extensions import Protocol

from src.core.context import Context


class ChannelMessage(Protocol):

    async def send_and_receive(
            self,
            context: Context,
            message: str,
    ) -> str | None:
        ...

    async def receive(
            self,
            context: Context,
            message: str,
    ) -> None:
        ...
