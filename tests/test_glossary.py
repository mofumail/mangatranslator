"""Glossary file and the NEW TERMS block of a translation response."""

from mangatl.translate.glossary import format_glossary_section, load_terms, record_new_terms, split_new_terms


def test_split_new_terms():
    text = "1: Hello\n2: Bye\n\nNEW TERMS:\n- 相川仁 -> Jin Aikawa\n* 梨花 → Rika"
    head, pairs = split_new_terms(text)
    assert head == "1: Hello\n2: Bye"
    assert pairs == [("相川仁", "Jin Aikawa"), ("梨花", "Rika")]
    assert split_new_terms("1: New terms apply here") == ("1: New terms apply here", [])


def test_record_new_terms_only_from_the_page(tmp_path):
    path = tmp_path / "glossary.txt"
    added, rejected = record_new_terms(
        path, [("猫猫(マオマオ)", "Maomao"), ("壬氏", "Jinshi")], ["猫猫は言った"]
    )
    assert added == [("猫猫", "Maomao")] and rejected == [("壬氏", "Jinshi")]
    assert load_terms(path) == {"猫猫": "Maomao"}
    assert record_new_terms(path, [("猫猫", "Mao")], ["猫猫"]) == ([], [])  # first entry wins
    assert format_glossary_section(load_terms(path)) == "\n## GLOSSARY\n猫猫 -> Maomao\n"
