"""Per-language surface data: scripts, digits, month names, honorifics, name pools.

This module is pure data plus a few tiny accessors. It is deliberately separate from
the generator so that adding a language is a data change, not a code change.

The romanisation lists are the load-bearing part of this file. A model reading a
Devanagari document may legitimately answer either in the source script or in Latin
transliteration -- both are correct extractions, and a verifier that accepts only one
of them measures script preference rather than reading comprehension. Because we
*generate* the corpus, we know every acceptable spelling by construction, so the
verifier can accept a closed set instead of resorting to fuzzy string distance.
Several entries carry genuinely competing conventions in real use
(Chattopadhyay/Chatterjee, Bandyopadhyay/Banerjee, Verma/Varma) and all are accepted.

The month tables are fenced with `# fmt: off` so they stay laid out six-per-line. Left
to the formatter each language becomes a fourteen-line column, which makes a calendar
year impossible to scan and a missing or misordered month easy to miss in review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LanguageCode = Literal["hi", "mr", "ta", "bn"]

# Unicode decimal-digit blocks. Kept explicit rather than derived from unicodedata
# because the generator needs to *render* digits in a chosen script, which is the
# inverse of what unicodedata offers.
DIGITS: dict[str, str] = {
    "latn": "0123456789",
    "deva": "०१२३४५६७८९",
    "taml": "௦௧௨௩௪௫௬௭௮௯",
    "beng": "০১২৩৪৫৬৭৮৯",
}


@dataclass(frozen=True)
class Name:
    """One person's name in the source script plus every romanisation we will accept."""

    native: str
    roman: tuple[str, ...]


@dataclass(frozen=True)
class Language:
    code: LanguageCode
    name_en: str
    script: str
    """Key into DIGITS for this language's native digit set."""

    months: tuple[str, ...]
    honorifics: tuple[str, ...]
    given: tuple[Name, ...]
    surnames: tuple[Name, ...]
    currency: str
    """Abbreviated currency word in this script, e.g. 'रु.' / 'ரூ.' / 'টা.'."""

    lakh: str
    crore: str
    """Multiplier words in this script. Rendering a Tamil document with Devanagari
    multiplier words was a real bug: the generator must stay inside one script for the
    document's own furniture, even though code-mixed Latin is realistic."""

    labels: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Field labels as they appear in documents, e.g. 'due_date' -> ('देय तिथि', ...)."""


# fmt: off
HINDI_MONTHS = (
    "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
)
MARATHI_MONTHS = (
    "जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून",
    "जुलै", "ऑगस्ट", "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर",
)
TAMIL_MONTHS = (
    "ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்",
    "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்",
)
BENGALI_MONTHS = (
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
    "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
)
# fmt: on


HINDI = Language(
    code="hi",
    name_en="Hindi",
    script="deva",
    months=HINDI_MONTHS,
    honorifics=("श्री", "श्रीमती", "कुमारी", "डॉ."),
    currency="रु.",
    lakh="लाख",
    crore="करोड़",
    given=(
        Name("रमेश", ("Ramesh",)),
        Name("सुनीता", ("Sunita", "Suneeta")),
        Name("अनिल", ("Anil",)),
        Name("प्रिया", ("Priya",)),
        Name("विकास", ("Vikas", "Vikash")),
        Name("मीना", ("Meena", "Mina")),
        Name("राजेश", ("Rajesh",)),
        Name("कविता", ("Kavita", "Kavitha")),
    ),
    surnames=(
        Name("शर्मा", ("Sharma",)),
        Name("वर्मा", ("Verma", "Varma")),
        Name("गुप्ता", ("Gupta",)),
        Name("सिंह", ("Singh",)),
        Name("यादव", ("Yadav",)),
        Name("मिश्रा", ("Mishra", "Misra")),
    ),
    labels={
        "addressee": ("प्रति", "ग्राहक का नाम"),
        "payable": ("देय राशि", "कुल देय राशि"),
        "due_date": ("देय तिथि", "अंतिम तिथि"),
        "reference": ("संदर्भ क्रमांक", "संदर्भ संख्या"),
        "issue_date": ("जारी दिनांक",),
        "late_fee": ("विलंब शुल्क",),
        "arrears": ("पिछला बकाया",),
        "officer": ("जारीकर्ता अधिकारी",),
        "helpline": ("हेल्पलाइन",),
        "next_cycle": ("अगली देय तिथि",),
    },
)

MARATHI = Language(
    code="mr",
    name_en="Marathi",
    script="deva",
    months=MARATHI_MONTHS,
    honorifics=("श्री", "श्रीमती", "कु."),
    currency="रु.",
    lakh="लाख",
    crore="कोटी",
    given=(
        Name("सुरेश", ("Suresh",)),
        Name("स्वाती", ("Swati", "Swathi")),
        Name("प्रकाश", ("Prakash",)),
        Name("वैशाली", ("Vaishali",)),
        Name("नितीन", ("Nitin", "Niteen")),
        Name("मंगला", ("Mangala",)),
    ),
    surnames=(
        Name("देशमुख", ("Deshmukh",)),
        Name("पाटील", ("Patil", "Patel")),
        Name("जोशी", ("Joshi",)),
        Name("कुलकर्णी", ("Kulkarni",)),
        Name("साळुंखे", ("Salunkhe", "Salunke")),
    ),
    labels={
        "addressee": ("प्रति", "ग्राहकाचे नाव"),
        "payable": ("देय रक्कम", "एकूण देय रक्कम"),
        "due_date": ("देय दिनांक", "अंतिम दिनांक"),
        "reference": ("संदर्भ क्रमांक",),
        "issue_date": ("जारी दिनांक",),
        "late_fee": ("विलंब शुल्क",),
        "arrears": ("मागील थकबाकी",),
        "officer": ("अधिकारी",),
        "helpline": ("मदत क्रमांक",),
        "next_cycle": ("पुढील देय दिनांक",),
    },
)

TAMIL = Language(
    code="ta",
    name_en="Tamil",
    script="taml",
    months=TAMIL_MONTHS,
    honorifics=("திரு", "திருமதி", "செல்வி"),
    currency="ரூ.",
    lakh="லட்சம்",
    crore="கோடி",
    given=(
        Name("முருகன்", ("Murugan",)),
        Name("லட்சுமி", ("Lakshmi", "Lakshmy", "Laxmi")),
        Name("கார்த்திக்", ("Karthik", "Karthick", "Kartik")),
        Name("தமிழ்ச்செல்வி", ("Tamilselvi", "Thamizhselvi")),
        Name("ராஜா", ("Raja",)),
    ),
    surnames=(
        Name("சுப்ரமணியன்", ("Subramanian", "Subramaniam", "Subramanyan")),
        Name("கிருஷ்ணன்", ("Krishnan",)),
        Name("ராமன்", ("Raman",)),
        Name("நடராஜன்", ("Natarajan", "Nadarajan")),
    ),
    labels={
        "addressee": ("பெறுநர்", "வாடிக்கையாளர் பெயர்"),
        "payable": ("செலுத்த வேண்டிய தொகை",),
        "due_date": ("கடைசி தேதி", "செலுத்த வேண்டிய தேதி"),
        "reference": ("குறிப்பு எண்",),
        "issue_date": ("வழங்கிய தேதி",),
        "late_fee": ("தாமத கட்டணம்",),
        "arrears": ("முந்தைய நிலுவை",),
        "officer": ("அலுவலர்",),
        "helpline": ("உதவி எண்",),
        "next_cycle": ("அடுத்த தேதி",),
    },
)

BENGALI = Language(
    code="bn",
    name_en="Bengali",
    script="beng",
    months=BENGALI_MONTHS,
    honorifics=("শ্রী", "শ্রীমতী"),
    currency="টা.",
    lakh="লক্ষ",
    crore="কোটি",
    given=(
        Name("সুব্রত", ("Subrata", "Subrato")),
        Name("অনিতা", ("Anita",)),
        Name("দেবাশিস", ("Debashis", "Debasish", "Devashish")),
        Name("রুমা", ("Ruma",)),
        Name("প্রদীপ", ("Pradip", "Pradeep")),
    ),
    surnames=(
        # Both members of these pairs are in genuine everyday use for the same
        # surname; a verifier that accepts only one is simply wrong.
        Name("চট্টোপাধ্যায়", ("Chattopadhyay", "Chatterjee")),
        Name("বন্দ্যোপাধ্যায়", ("Bandyopadhyay", "Banerjee")),
        Name("মুখার্জি", ("Mukherjee", "Mukherji")),
        Name("ঘোষ", ("Ghosh",)),
        Name("দাস", ("Das",)),
    ),
    labels={
        "addressee": ("প্রাপক", "গ্রাহকের নাম"),
        "payable": ("প্রদেয় পরিমাণ",),
        "due_date": ("শেষ তারিখ", "প্রদেয় তারিখ"),
        "reference": ("রেফারেন্স নম্বর",),
        "issue_date": ("ইস্যু তারিখ",),
        "late_fee": ("বিলম্ব ফি",),
        "arrears": ("পূর্বের বকেয়া",),
        "officer": ("কর্মকর্তা",),
        "helpline": ("হেল্পলাইন",),
        "next_cycle": ("পরবর্তী তারিখ",),
    },
)

LANGUAGES: dict[LanguageCode, Language] = {
    lang.code: lang for lang in (HINDI, MARATHI, TAMIL, BENGALI)
}

# Romanised honorifics a model may prepend regardless of source script. Stripped from
# both sides before name comparison so that honorific choice never decides a score.
# fmt: off
ROMAN_HONORIFICS: tuple[str, ...] = (
    "shri", "sri", "smt", "smt.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.",
    "dr", "dr.", "thiru", "tmt", "selvi", "kumari", "km", "sushri",
)
# fmt: on
