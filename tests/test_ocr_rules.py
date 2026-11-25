"""OCR read rules: script detection, list detection, picking between two reads."""

from types import SimpleNamespace

from mangatl.ocr.line_ocr import is_japanese, is_list, is_other_cjk, pick_read, pick_sfx_read, script_ratios


def test_script_ratios():
    n, kana, han = script_ratios("お前には夏休み")
    assert n == 7 and kana > 0.4 and han > 0.2
    assert script_ratios("!!...") == (0, 0.0, 0.0)


def test_is_japanese_needs_kana():
    assert is_japanese("お前には夏休みの間")
    assert not is_japanese("夏休")  # too short
    assert not is_japanese("Hello world")


def test_chinese_block_beside_japanese():
    assert is_other_cjk("我们这里说话时没有问题吗")  # simplified-only characters, no kana
    assert not is_other_cjk("状態：乳房開発完了 人格消失")  # all-kanji Japanese status screen
    assert not is_other_cjk("お前には夏休みの間")


def test_list_needs_horizontal_labeled_lines():
    horizontal = [SimpleNamespace(vertical=False)] * 3
    assert is_list(horizontal, ["Yuki: hi", "Take: cute", "Mimi: lol"])
    assert not is_list(horizontal, ["one line", "and another", "Yuki: hi"])
    assert not is_list([SimpleNamespace(vertical=True)] * 2, ["Yuki: hi", "Take: cute"])


def test_pick_read_keeps_the_fuller_read():
    assert pick_read("ドクッ", "ド") == "ドクッ"  # one stopped early
    assert pick_read("十年後", "十年後ーー夏") == "十年後ーー夏"
    assert pick_read("はい、", "はい") == "はい、"  # same letters: keep the punctuation
    assert pick_read("むり", "むに") == "むに"  # disagree: the preferred read
    assert pick_read("abc", "   ") == "abc"
    assert pick_read("...", "えっ") == "えっ"  # a read of only dots is not "contained"


def test_pick_sfx_read_prefers_the_padded_crop():
    assert pick_sfx_read("むり", "むに") == "むに"
    assert pick_sfx_read("ドクッ", "ド") == "ドクッ"
