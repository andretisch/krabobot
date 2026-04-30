import json

from krabobot.channels.vk import VKChannel


def test_extract_saved_doc_from_list_payload() -> None:
    payload = [{"id": 10, "owner_id": 20, "access_key": "abc"}]
    doc = VKChannel._extract_saved_doc(payload)
    assert doc is not None
    assert doc["id"] == 10
    assert doc["owner_id"] == 20


def test_extract_saved_doc_from_dict_doc_payload() -> None:
    payload = {"type": "doc", "doc": {"id": 11, "owner_id": 21}}
    doc = VKChannel._extract_saved_doc(payload)
    assert doc is not None
    assert doc["id"] == 11
    assert doc["owner_id"] == 21


def test_extract_saved_doc_from_dict_audio_message_payload() -> None:
    payload = {"type": "audio_message", "audio_message": {"id": 12, "owner_id": 22}}
    doc = VKChannel._extract_saved_doc(payload)
    assert doc is not None
    assert doc["id"] == 12
    assert doc["owner_id"] == 22


def test_extract_upload_url_from_nested_response_payload() -> None:
    payload = {"response": {"upload_url": "https://example/upload"}}
    assert VKChannel._extract_upload_url(payload) == "https://example/upload"


def test_extract_saved_doc_from_wrapped_response_list() -> None:
    payload = {"response": [{"id": 13, "owner_id": 23}]}
    doc = VKChannel._extract_saved_doc(payload)
    assert doc is not None
    assert doc["id"] == 13
    assert doc["owner_id"] == 23


def test_vk_plain_text_strips_common_markdown() -> None:
    raw = "**bold** and `code` and [link](https://example.com)"
    txt = VKChannel._vk_plain_text(raw)
    assert "**" not in txt
    assert "`" not in txt
    assert "bold" in txt
    assert "code" in txt
    assert "link (https://example.com)" in txt


def test_vk_commands_keyboard_contains_basic_commands() -> None:
    keyboard = json.loads(VKChannel._vk_commands_keyboard())
    labels = []
    for row in keyboard["buttons"]:
        for button in row:
            labels.append(button["action"]["label"])
    assert "/help" in labels
    assert "/id" in labels
    assert "/link" in labels
    assert "/status" in labels
    assert "/new" in labels
    assert "/stop" in labels
    assert "/restart" in labels
    assert "/tts status" in labels
