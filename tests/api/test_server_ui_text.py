from krabobot.api.server import _ui_text_from_stored_content, derive_session_title


def test_ui_text_strips_linked_accounts_from_stored_history() -> None:
    linked = (
        "[Linked Accounts]\n"
        "user_id: u1\n"
        "accounts: telegram:123"
    )
    content = f"{linked}\n\nHello again"

    assert _ui_text_from_stored_content(content) == "Hello again"


def test_derive_session_title_from_first_user_message() -> None:
    messages = [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "Купить молоко и хлеб завтра утром пожалуйста очень длинный текст"},
        {"role": "user", "content": "второе сообщение"},
    ]
    title = derive_session_title(messages)
    assert title.startswith("Купить молоко")
    assert "второе" not in title
    assert len(title) <= 50


def test_derive_session_title_prefers_metadata() -> None:
    messages = [{"role": "user", "content": "оригинальный текст"}]
    assert derive_session_title(messages, {"title": "Мой диалог"}) == "Мой диалог"


def test_derive_session_title_empty_without_user() -> None:
    assert derive_session_title([]) == ""
    assert derive_session_title([{"role": "assistant", "content": "only bot"}]) == ""


def test_derive_session_title_strips_linked_accounts() -> None:
    linked = (
        "[Linked Accounts]\n"
        "user_id: u1\n"
        "accounts: telegram:123\n\n"
        "Реальный вопрос пользователя"
    )
    assert derive_session_title([{"role": "user", "content": linked}]) == (
        "Реальный вопрос пользователя"
    )
