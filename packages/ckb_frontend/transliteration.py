"""
Sorani Transliteration & Pronunciation Override Engine.
Handles English/Arabic loanwords, tech terminology, and speaker-specific pronunciation lexicons.
"""

import re
from typing import Dict, Optional


class SoraniTransliteration:
    """
    Transliterates Latin/English tech terms, brands, and foreign names into Sorani orthography.
    Also handles custom speaker-specific pronunciation dictionaries.
    """

    # Common English words and tech terminology mapped to standard Sorani spoken form
    TECH_LEXICON: Dict[str, str] = {
        "ai": "ئەی ئای",
        "api": "ئەی پی ئای",
        "app": "ئەپ",
        "audio": "ئۆدیۆ",
        "bluetooth": "بلووتووس",
        "cpu": "سی پی یوو",
        "database": "داتابەیس",
        "docker": "دۆکەر",
        "email": "ئیمەیڵ",
        "facebook": "فەیسبووک",
        "fastapi": "فاست ئەی پی ئای",
        "github": "گیتھەب",
        "google": "گووگڵ",
        "gpu": "جی پی یوو",
        "instagram": "ئینستاگرام",
        "internet": "ئینتەرنێت",
        "linux": "لینۆکس",
        "lora": "لۆڕا",
        "microsoft": "مایکرۆسۆفت",
        "model": "مۆدێل",
        "nextjs": "نێکست جەی ئێس",
        "online": "ئۆنلاین",
        "pdf": "پی دی ئێف",
        "podcast": "پۆدکاست",
        "python": "پایسۆن",
        "server": "سێرڤەر",
        "studio": "ستۆدیۆ",
        "tiktok": "تیکتۆک",
        "tts": "تی تی ئێس",
        "url": "یوو ئاڕ ئێڵ",
        "vllm": "ڤی ئێڵ ئێڵ ئێم",
        "voxcpm": "ڤۆکس سی پی ئێم",
        "wav": "واڤ",
        "website": "ماڵپەڕ",
        "whatsapp": "واتسئەپ",
        "wi-fi": "وای فای",
        "wifi": "وای فای",
        "windows": "ویندۆز",
        "youtube": "یووتیووب",
    }

    # Latin to Kurdish phonetic character mapping
    LATIN_TO_SORANI: Dict[str, str] = {
        'a': 'ا',
        'b': 'ب',
        'c': 'ج',
        'ch': 'چ',
        'd': 'د',
        'e': 'ە',
        'ê': 'ێ',
        'f': 'ف',
        'g': 'گ',
        'h': 'ه',
        'i': 'ی',
        'î': 'ی',
        'j': 'ژ',
        'k': 'ک',
        'l': 'ل',
        'll': 'ڵ',
        'm': 'م',
        'n': 'ن',
        'o': 'ۆ',
        'p': 'پ',
        'q': 'ق',
        'r': 'ر',
        'rr': 'ڕ',
        's': 'س',
        'sh': 'ش',
        't': 'ت',
        'u': 'و',
        'û': 'وو',
        'v': 'ڤ',
        'w': 'و',
        'x': 'خ',
        'y': 'ی',
        'z': 'ز',
    }

    def __init__(self, custom_lexicon: Optional[Dict[str, str]] = None):
        self.custom_lexicon = custom_lexicon or {}

    def apply_overrides(self, text: str) -> str:
        """Apply custom pronunciation dictionary overrides."""
        # 1. Custom speaker dictionary overrides (highest priority)
        for term, spoken in self.custom_lexicon.items():
            pattern = r'(?i)\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, spoken, text)

        # 2. Common tech lexicon
        for term, spoken in self.TECH_LEXICON.items():
            pattern = r'(?i)\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, spoken, text)

        return text
