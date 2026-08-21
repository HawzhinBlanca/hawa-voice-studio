"""
Unit Tests for Kurdish Sorani Normalizer & G2P Phoneme QA (packages/ckb-frontend).
"""

import pytest
from packages.ckb_frontend import SoraniNormalizer, SoraniPhonemeQA, SoraniTransliteration


@pytest.fixture
def normalizer():
    return SoraniNormalizer()


def test_character_folding(normalizer):
    """Test Arabic kaf, yeh, and heh variants fold into standard Kurdish characters."""
    # Arabic kaf (0643) -> Kurdish kaf (06A9)
    # Arabic yeh (064A) -> Kurdish yeh (06CC)
    # Arabic teh marbuta (0629) -> Kurdish ae (06D5)
    raw = "كۆماری كوردستان لە ساڵی ١٩٤٦"
    normalized = normalizer.normalize(raw)
    
    assert "ك" not in normalized
    assert "ک" in normalized
    assert "ی" in normalized


def test_eastern_arabic_numbers(normalizer):
    """Test Eastern Arabic digits expansion to written Sorani words."""
    assert normalizer.normalize("٥") == "پێنج"
    assert normalizer.normalize("١٢") == "دوازدە"
    assert normalizer.normalize("٧٥") == "حەفتا و پێنج"
    assert normalizer.normalize("١٠٠") == "سەد"
    assert normalizer.normalize("١٥٠") == "سەد و پەنجا"
    assert normalizer.normalize("١٠٠٠") == "هەزار"
    assert normalizer.normalize("٢٥٠٠") == "دوو هەزار و پێنجسەد"
    assert normalizer.normalize("١٠٠٠٠٠٠") == "یەک ملیۆن"


def test_currency_expansion(normalizer):
    """Test USD, IQD, and other currency symbol expansions."""
    res_usd = normalizer.normalize("$150")
    assert "دۆلار" in res_usd
    assert "سەد و پەنجا" in res_usd

    res_iqd = normalizer.normalize("250000 IQD")
    assert "دیناری عێراقی" in res_iqd
    assert "دووسەد و پەنجا هەزار" in res_iqd


def test_percentage_expansion(normalizer):
    """Test percentage conversion."""
    res = normalizer.normalize("25%")
    assert "لە سەدا بیست و پێنج" in res


def test_date_and_time_expansion(normalizer):
    """Test Gregorian & Solar date expansion and time formatting."""
    res_date = normalizer.normalize("2026/08/21")
    assert "ئاب" in res_date or "ئابی" in res_date
    assert "دوو هەزار و بیست و شەش" in res_date

    res_time = normalizer.normalize("14:30")
    assert "کاتژمێر چواردە و نیو" in res_time


def test_phoneme_qa_coverage():
    """Test phonemic coverage analyzer."""
    qa = SoraniPhonemeQA()
    text = "ڕۆڵەی دڵسۆزی نیشتمان دەنگێکی زوڵاڵی هەیە."
    report = qa.analyze_coverage(text)
    
    assert report["total_characters"] > 0
    assert "ɫ" in report["phoneme_frequencies"]  # velarized ڵ
    assert "r" in report["phoneme_frequencies"]  # trill ڕ
    assert report["coverage_ratio"] > 0.3
