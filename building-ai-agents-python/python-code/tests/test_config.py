from lib.config import as_text


class Message:
    def __init__(self, content):
        self.content = content


def test_as_text_with_plain_string():
    assert as_text(Message("hello")) == "hello"


def test_as_text_with_content_blocks():
    blocks = [{"type": "text", "text": "Hello, "}, {"type": "other"}, "world"]
    assert as_text(Message(blocks)) == "Hello, world"
