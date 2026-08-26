from krabobot.api.server import _ui_text_from_stored_content


def test_ui_text_strips_linked_accounts_from_stored_history() -> None:
    linked = (
        "[Linked Accounts]\n"
        "user_id: u1\n"
        "accounts: telegram:123"
    )
    content = f"{linked}\n\nHello again"

    assert _ui_text_from_stored_content(content) == "Hello again"
