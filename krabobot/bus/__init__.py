"""Message bus module for decoupled channel-agent communication."""

from krabobot.bus.events import InboundMessage, OutboundMessage
from krabobot.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
