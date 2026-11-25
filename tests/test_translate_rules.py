"""Translation post-processing: truncation retry, failed / skipped items, short-item selection."""

from mangatl.translate.page import _mark_failed_and_skipped, _retry_truncated
from mangatl.translate.vision_pass import _short_item, _words

LONG = "これはとても長い文章で、途中で切れてしまうと困るのですが、本当に長いのです。"


def test_retry_truncated_keeps_the_fuller_answer():
    calls = []

    def translate(ocr_out):
        calls.append(ocr_out)
        return ["This is a very long sentence that must not be cut off halfway, it really is long.", "Hi"]

    meta = [{"ocr_text": LONG}, {"ocr_text": "やあ"}]
    texts = _retry_truncated(["-A little while later.", "Hello"], meta, [], translate)
    assert texts[0].startswith("This is a very long") and texts[1] == "Hello"  # only the bad item changes
    assert calls == [[]]  # retried once, with a fresh OCR list


def test_retry_truncated_leaves_good_pages_alone():
    def translate(ocr_out):
        raise AssertionError("must not retry")

    assert _retry_truncated(["Hi"], [{"ocr_text": "やあ"}], [], translate) == ["Hi"]


def test_missing_items_fail_and_skipped_blocks_erase():
    texts = ["[OpenAI-Compatible-Translate: missing item 2]", "我们", "Fine"]
    _mark_failed_and_skipped(texts, [{}, {"mt_skip": True}, {}])
    assert texts == ["[OCR FAILED]", "", "Fine"]


def test_short_items():
    assert _short_item("[OSB] ドドドッ", osb=True)
    assert _short_item("[Bubble] えっ!?", osb=False)
    assert not _short_item("[Bubble] お前には夏休みの間", osb=False)  # longer than 5 letters
    assert not _short_item("[OCR FAILED]", osb=True)


def test_words_ignores_restyling():
    assert _words("*Huff*") == _words("**HUFF**")
    assert _words("Huff") != _words("Puff")
