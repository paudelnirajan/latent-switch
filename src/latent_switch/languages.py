"""Language registry. Codes are FLORES-style `{ISO 639-3}_{ISO 15924}`, e.g. npi_Deva = Nepali in Devanagari."""

# Language names written in that language (endonyms); used in prompts so they contain no English.
ENDONYMS = {
    "eng_Latn": "English",
    "fra_Latn": "Français",
    "deu_Latn": "Deutsch",
    "rus_Cyrl": "Русский",
    "cmn_Hans": "中文",
    "npi_Deva": "नेपाली",
    "hin_Deva": "हिन्दी",
}

# ISO 15924 script code → Unicode script name as used by the `regex` package (\p{Han}, \p{Latin}, ...)
UNICODE_SCRIPT = {
    "Latn": "Latin",
    "Cyrl": "Cyrillic",
    "Hans": "Han",
    "Hant": "Han",
    "Deva": "Devanagari",
}


def script_of(code: str) -> str:
    """'npi_Deva' → 'Deva'."""
    return code.split("_")[1]


def endonym(code: str) -> str:
    """The language's name in its own language; fails loudly for unregistered languages."""
    if code not in ENDONYMS:
        raise KeyError(f"no endonym registered for {code!r}; add it to latent_switch.languages.ENDONYMS")
    return ENDONYMS[code]
