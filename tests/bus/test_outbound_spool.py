"""Tests for cross-process outbound spool (serve → gateway)."""

from __future__ import annotations

from pathlib import Path

from krabobot.bus.events import OutboundMessage
from krabobot.bus.spool import (
    claim_spooled_outbound,
    enqueue_outbound,
    is_local_outbound_channel,
    outbound_spool_dir,
    route_serve_outbound,
)


def test_is_local_outbound_channel() -> None:
    assert is_local_outbound_channel("api")
    assert is_local_outbound_channel("CLI")
    assert is_local_outbound_channel("voice")
    assert is_local_outbound_channel("system")
    assert is_local_outbound_channel("")
    assert not is_local_outbound_channel("email")
    assert not is_local_outbound_channel("telegram")
    assert not is_local_outbound_channel("vk")


def test_route_serve_outbound_drops_local_spools_email() -> None:
    assert route_serve_outbound(
        OutboundMessage(channel="api", chat_id="sess", content="hi")
    ) is None
    routed = route_serve_outbound(
        OutboundMessage(channel="email", chat_id="a@b.c", content="mail")
    )
    assert routed is not None
    assert routed.metadata.get("force_send") is True
    assert routed.chat_id == "a@b.c"


def test_enqueue_and_claim_roundtrip(tmp_path: Path) -> None:
    msg = OutboundMessage(
        channel="email",
        chat_id="user@example.com",
        content="hello from web",
        media=["/tmp/a.pdf"],
        metadata={"force_send": True, "subject": "Hi"},
    )
    path = enqueue_outbound(tmp_path, msg)
    assert path.exists()
    assert path.parent == outbound_spool_dir(tmp_path)

    claimed = claim_spooled_outbound(tmp_path)
    assert len(claimed) == 1
    out = claimed[0]
    assert out.channel == "email"
    assert out.chat_id == "user@example.com"
    assert out.content == "hello from web"
    assert out.media == ["/tmp/a.pdf"]
    assert out.metadata.get("force_send") is True
    assert out.metadata.get("subject") == "Hi"

    # Spool file consumed
    assert not path.exists()
    assert claim_spooled_outbound(tmp_path) == []


def test_claim_skips_corrupt_and_continues(tmp_path: Path) -> None:
    spool = outbound_spool_dir(tmp_path)
    bad = spool / "1_bad.json"
    bad.write_text("not-json", encoding="utf-8")
    good = OutboundMessage(channel="email", chat_id="a@b.c", content="ok")
    enqueue_outbound(tmp_path, good)

    claimed = claim_spooled_outbound(tmp_path)
    assert len(claimed) == 1
    assert claimed[0].content == "ok"
    assert list(spool.glob("*.failed.json"))


def test_channel_manager_claims_spool_into_pending(tmp_path: Path) -> None:
    from krabobot.channels.manager import ChannelManager
    from krabobot.config.schema import Config

    enqueue_outbound(
        tmp_path,
        OutboundMessage(channel="email", chat_id="x@y.z", content="spooled"),
    )
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    mgr = ChannelManager.__new__(ChannelManager)
    mgr.config = cfg
    claimed = mgr._claim_spooled_outbound()
    assert len(claimed) == 1
    assert claimed[0].chat_id == "x@y.z"
