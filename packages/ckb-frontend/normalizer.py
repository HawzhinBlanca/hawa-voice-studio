"""
Central Kurdish (Sorani / ckb) Text Frontend & Normalizer
Production-grade normalization for Kurdish TTS (VoxCPM2 & CosyVoice3).
"""

import re
from typing import Dict, List, Optional, Tuple


class SoraniNormalizer:
    """
    Production-grade Kurdish Sorani (Central Kurdish / کوردیی ناوەندی) Normalizer.
    
    Handles:
    1. Unicode & character folding (Arabic/Persian/Kurdish character variants)
    2. Zero-width non-joiner (ZWNJ / نیم‌بۆشایی) and tatweel cleanup
    3. Eastern Arabic & Persian numerals expansion to written Sorani words
    4. Currency expansion (IQD, USD, EUR, etc.)
    5. Date & time expansion with Kurdish clitic/suffix handling (e.g., ١٤:٣٠دا)
    6. Phone numbers, percentages, abbreviations, and symbols
    7. Foreign loanword & English/Arabic name transliteration preparation
    8. Sorani spelling standardization & prosody punctuation
    """

    CHAR_MAP: Dict[str, str] = {
        '\u0643': '\u06a9',  # ك -> ک
        '\u064a': '\u06cc',  # ي -> ی
        '\u0649': '\u06cc',  # ى -> ی
        '\u0629': '\u06d5',  # ة -> ە
        '\u06c0': '\u06d5',  # ۀ -> ە
        '\u0622': '\u0626\u0627',  # آ -> ئا
        '\u0623': '\u0626\u06d5',  # أ -> ئە
        '\u0625': '\u0626\u06cc',  # إ -> ئی
        '\u0624': '\u0626\u06c6',  # ؤ -> ئۆ
        '\u06b5': '\u06b5',  # ڵ
        '\u0695': '\u0695',  # ڕ
        '\u06d5': '\u06d5',  # ە
        '\u06c6': '\u06c6',  # ۆ
        '\u06ce': '\u06ce',  # ێ
        '\u06c7': '\u06c7',  # ۇ
    }

    DIGIT_MAP: Dict[str, str] = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
        '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
        '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
    }

    ONES: Dict[int, str] = {
        0: "سفر",
        1: "یەک",
        2: "دوو",
        3: "سێ",
        4: "چوار",
        5: "پێنج",
        6: "شەش",
        7: "حەوت",
        8: "هەشت",
        9: "نۆ",
    }

    TEENS: Dict[int, str] = {
        10: "دە",
        11: "یازدە",
        12: "دوازدە",
        13: "سێزدە",
        14: "چواردە",
        15: "پازدە",
        16: "شازدە",
        17: "حەڤدە",
        18: "هەژدە",
        19: "نۆزدە",
    }

    TENS: Dict[int, str] = {
        20: "بیست",
        30: "سی",
        40: "چل",
        50: "پەنجا",
        60: "شەست",
        70: "حەفتا",
        80: "هەشتا",
        90: "نەوەد",
    }

    HUNDREDS: Dict[int, str] = {
        100: "سەد",
        200: "دووسەد",
        300: "سێسەد",
        400: "چوارسەد",
        500: "پێنجسەد",
        600: "شەشسەد",
        700: "حەوتسەد",
        800: "هەشتسەد",
        900: "نۆسەد",
    }

    SCALES: List[Tuple[int, str]] = [
        (1_000_000_000_000, "تریلیۆن"),
        (1_000_000_000, "ملیار"),
        (1_000_000, "ملیۆن"),
        (1_000, "هەزار"),
    ]

    MONTHS_SOLAR: Dict[int, str] = {
        1: "نەورۆز",
        2: "گوڵان",
        3: "جۆزەردان",
        4: "پووشپەڕ",
        5: "گەلاوێژ",
        6: "خەرمانان",
        7: "بەران",
        8: "خەزەڵوەر",
        9: "سەرماوەز",
        10: "بەفرانبار",
        11: "ڕێبەندان",
        12: "ڕەشەمە",
    }

    MONTHS_GREGORIAN: Dict[int, str] = {
        1: "کانوونی دووەم",
        2: "شوبات",
        3: "ئازار",
        4: "نیسان",
        5: "ئایار",
        6: "حوزەیران",
        7: "تەمووز",
        8: "ئاب",
        9: "ئەیلوول",
        10: "تشرینی یەکەم",
        11: "تشرینی دووەم",
        12: "کانوونی یەکەم",
    }

    CURRENCIES: Dict[str, Tuple[str, str]] = {
        "$": ("دۆلار", "سەنت"),
        "USD": ("دۆلار", "سەنت"),
        "د.ع": ("دیناری عێراقی", "فلس"),
        "IQD": ("دیناری عێراقی", "فلس"),
        "€": ("یۆرۆ", "سەنت"),
        "EUR": ("یۆرۆ", "سەنت"),
        "£": ("پاوەند", "پێنس"),
        "GBP": ("پاوەند", "پێنس"),
        "تۆمان": ("تۆمان", ""),
        "ڕیاڵ": ("ڕیاڵ", ""),
    }

    ABBREVIATIONS: Dict[str, str] = {
        "د.": "دکتۆر",
        "پ.": "پڕۆفیسۆر",
        "ئـ.": "ئەندازیار",
        "ک.م": "کیلۆمەتر",
        "کم": "کیلۆمەتر",
        "سم": "سانتیمەتر",
        "م.": "مەتر",
        "کگم": "کیلۆگرام",
        "ت.ب": "تێبینی",
        "ز.": "زاینی",
        "ک.": "کۆچی",
    }

    def __init__(self, use_gregorian_months: bool = True):
        self.use_gregorian_months = use_gregorian_months

    def fold_characters(self, text: str) -> str:
        for src, dst in self.CHAR_MAP.items():
            text = text.replace(src, dst)
        text = text.replace('\u0640', '')
        text = re.sub(r'[\u064b\u064c\u064d\u064e\u064f\u0650\u0652\u0670]', '', text)
        text = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', text)
        text = re.sub(r'\u200c{2,}', '\u200c', text)
        return text

    def normalize_digits(self, text: str) -> str:
        res = []
        for ch in text:
            res.append(self.DIGIT_MAP.get(ch, ch))
        return "".join(res)

    def number_to_words(self, n: int) -> str:
        if n < 0:
            return "مایس " + self.number_to_words(abs(n))
        if n < 10:
            return self.ONES[n]
        if n < 20:
            return self.TEENS[n]
        if n < 100:
            tens_val = (n // 10) * 10
            rem = n % 10
            if rem == 0:
                return self.TENS[tens_val]
            return f"{self.TENS[tens_val]} و {self.ONES[rem]}"
        if n < 1000:
            hundreds_val = (n // 100) * 100
            rem = n % 100
            if rem == 0:
                return self.HUNDREDS[hundreds_val]
            return f"{self.HUNDREDS[hundreds_val]} و {self.number_to_words(rem)}"
        
        for scale_val, scale_name in self.SCALES:
            if n >= scale_val:
                lead = n // scale_val
                rem = n % scale_val
                lead_str = self.number_to_words(lead)
                if scale_val == 1000 and lead == 1:
                    base = "هەزار"
                else:
                    base = f"{lead_str} {scale_name}"
                
                if rem == 0:
                    return base
                return f"{base} و {self.number_to_words(rem)}"
        
        return str(n)

    def expand_decimal(self, match: re.Match) -> str:
        whole = int(match.group(1))
        fraction = match.group(2)
        whole_str = self.number_to_words(whole)
        
        frac_words = []
        for digit in fraction:
            frac_words.append(self.ONES[int(digit)])
        return f"{whole_str} پۆینت {' '.join(frac_words)}"

    def expand_currencies(self, text: str) -> str:
        def _curr_prefix(m: re.Match) -> str:
            sym = m.group(1)
            num_str = m.group(2).replace(',', '')
            suffix = m.group(3) or ""
            curr_name, sub_name = self.CURRENCIES.get(sym, (sym, ""))
            if '.' in num_str:
                parts = num_str.split('.')
                whole = int(parts[0])
                cents = int(parts[1][:2].ljust(2, '0'))
                res = f"{self.number_to_words(whole)} {curr_name}"
                if cents > 0 and sub_name:
                    res += f" و {self.number_to_words(cents)} {sub_name}"
                return res + suffix
            else:
                return f"{self.number_to_words(int(num_str))} {curr_name}{suffix}"

        text = re.sub(r'([\$€£])\s*(\d+(?:,\d{3})*(?:\.\d+)?)([\u0600-\u06FF]*)', _curr_prefix, text)

        def _curr_suffix(m: re.Match) -> str:
            num_str = m.group(1).replace(',', '')
            sym = m.group(2)
            curr_name, _ = self.CURRENCIES.get(sym, (sym, ""))
            return f"{self.number_to_words(int(num_str))} {curr_name}"

        text = re.sub(r'(\d+(?:,\d{3})*)\s*(د\.ع|IQD|USD|EUR|GBP|تۆمان|ڕیاڵ)', _curr_suffix, text)
        return text

    def expand_percentages(self, text: str) -> str:
        def _pct(m: re.Match) -> str:
            val = int(m.group(1) if m.group(1) else m.group(2))
            return f"لە سەدا {self.number_to_words(val)}"

        text = re.sub(r'%(\d+)|(\d+)%', _pct, text)
        return text

    def expand_dates_and_times(self, text: str) -> str:
        def _time(m: re.Match) -> str:
            hour = int(m.group(1))
            minute = int(m.group(2))
            suffix = m.group(3) or ""
            hour_word = self.number_to_words(hour)
            if minute == 0:
                base = f"کاتژمێر {hour_word}"
            elif minute == 15:
                base = f"کاتژمێر {hour_word} و چارەک"
            elif minute == 30:
                base = f"کاتژمێر {hour_word} و نیو"
            elif minute == 45:
                next_hour = self.number_to_words((hour % 24) + 1)
                base = f"کاتژمێر {next_hour} چارەکی کەم"
            else:
                min_word = self.number_to_words(minute)
                base = f"کاتژمێر {hour_word} و {min_word} خولەک"

            if suffix:
                return f"{base}{suffix}"
            return base

        text = re.sub(r'(\d{1,2}):(\d{2})([\u0600-\u06FF]*)', _time, text)

        def _date(m: re.Match) -> str:
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3))
            suffix = m.group(4) or ""
            
            month_dict = self.MONTHS_GREGORIAN if self.use_gregorian_months else self.MONTHS_SOLAR
            month_name = month_dict.get(month, f"مانگی {self.number_to_words(month)}")
            
            day_str = self.number_to_words(day)
            year_str = self.number_to_words(year)
            return f"{day_str}ی {month_name}ی ساڵی {year_str}{suffix}"

        text = re.sub(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})([\u0600-\u06FF]*)', _date, text)
        return text

    def expand_abbreviations(self, text: str) -> str:
        for abbr, expanded in self.ABBREVIATIONS.items():
            pattern = r'(?<!\w)' + re.escape(abbr) + r'(?!\w)'
            text = re.sub(pattern, expanded, text)
        return text

    def expand_plain_numbers(self, text: str) -> str:
        text = re.sub(r'\b(\d+)\.(\d+)\b', self.expand_decimal, text)
        
        def _comma_num(m: re.Match) -> str:
            num = int(m.group(0).replace(',', ''))
            return self.number_to_words(num)
        
        text = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', _comma_num, text)

        def _plain_num(m: re.Match) -> str:
            return self.number_to_words(int(m.group(0)))

        text = re.sub(r'\b\d+\b', _plain_num, text)
        return text

    def standardize_spelling_and_prosody(self, text: str) -> str:
        text = re.sub(r'\bدە\s+([کتپچڕسشخفڤقغحعھهلمنەیێۆۇڵڕگدژز])', r'دە\1', text)
        text = text.replace('?', '؟').replace(',', '،').replace(';', '؛')
        text = re.sub(r'([!؟،؛.])\1+', r'\1', text)
        text = re.sub(r'([!؟،؛.])(?=[^\s!؟،؛.])', r'\1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        
        text = self.fold_characters(text)
        text = self.normalize_digits(text)
        text = self.expand_abbreviations(text)
        text = self.expand_dates_and_times(text)
        text = self.expand_currencies(text)
        text = self.expand_percentages(text)
        text = self.expand_plain_numbers(text)
        text = self.standardize_spelling_and_prosody(text)
        return text


class SoraniPhonemeQA:
    CHAR_TO_IPA: Dict[str, str] = {
        'ئ': 'ʔ', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ج': 'd͡ʒ',
        'چ': 't͡ʃ', 'ح': 'ħ', 'خ': 'x', 'د': 'd', 'ر': 'ɾ',
        'ڕ': 'r', 'ز': 'z', 'ژ': 'ʒ', 'س': 's', 'ش': 'ʃ',
        'ع': 'ʕ', 'غ': 'ɣ', 'ف': 'f', 'ڤ': 'v', 'ق': 'q',
        'ک': 'k', 'گ': 'g', 'ل': 'l', 'ڵ': 'ɫ', 'م': 'm',
        'ن': 'n', 'و': 'w', 'ۆ': 'oː', 'ۇ': 'u', 'ھ': 'h',
        'ه': 'h', 'ە': 'a', 'ی': 'j', 'ێ': 'eː', 'ا': 'ɑː',
    }

    def analyze_coverage(self, text: str) -> Dict[str, any]:
        normalized = SoraniNormalizer().normalize(text)
        char_counts = {}
        phoneme_counts = {}
        
        for ch in normalized:
            if ch in self.CHAR_TO_IPA:
                char_counts[ch] = char_counts.get(ch, 0) + 1
                ipa = self.CHAR_TO_IPA[ch]
                phoneme_counts[ipa] = phoneme_counts.get(ipa, 0) + 1
        
        missing_chars = [ch for ch in self.CHAR_TO_IPA.keys() if ch not in char_counts]
        
        return {
            "total_characters": len(normalized),
            "unique_characters": len(char_counts),
            "character_frequencies": char_counts,
            "phoneme_frequencies": phoneme_counts,
            "missing_characters": missing_chars,
            "coverage_ratio": (len(self.CHAR_TO_IPA) - len(missing_chars)) / len(self.CHAR_TO_IPA),
        }
