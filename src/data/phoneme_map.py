"""
Phoneme mappings for decoding logits/IDs to characters.

LOGIT_TO_PHONEME follows an ARPAbet-style list with index 0 as BLANK.
PHONEME_TO_CHAR_MAP maps phoneme IDs to rough grapheme strings so that
phonemes_to_text can emit readable text for WER/CER computation.
"""

LOGIT_TO_PHONEME = [
    "BLANK",
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B",  "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH",
    " | ",
]


def _arpabet_to_grapheme():
    """
    Build a simple phoneme_id -> grapheme mapping.
    This is approximate; refine if you have a better lexicon.
    """
    mapping = {
        "BLANK": "",
        "AA": "a",
        "AE": "a",
        "AH": "u",
        "AO": "aw",
        "AW": "ow",
        "AY": "i",
        "B": "b",
        "CH": "ch",
        "D": "d",
        "DH": "th",
        "EH": "e",
        "ER": "er",
        "EY": "ay",
        "F": "f",
        "G": "g",
        "HH": "h",
        "IH": "i",
        "IY": "ee",
        "JH": "j",
        "K": "k",
        "L": "l",
        "M": "m",
        "N": "n",
        "NG": "ng",
        "OW": "o",
        "OY": "oy",
        "P": "p",
        "R": "r",
        "S": "s",
        "SH": "sh",
        "T": "t",
        "TH": "th",
        "UH": "u",
        "UW": "oo",
        "V": "v",
        "W": "w",
        "Y": "y",
        "Z": "z",
        "ZH": "zh",
        " | ": " ",
    }

    return {idx: mapping.get(ph, "?") for idx, ph in enumerate(LOGIT_TO_PHONEME)}


# Default approximate mapping
PHONEME_TO_CHAR_MAP = _arpabet_to_grapheme()
