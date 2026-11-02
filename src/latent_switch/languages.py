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
    "jpn_Jpan": "日本語",
    "arb_Arab": "العربية",
    "swh_Latn": "Kiswahili",
}

# ISO 15924 script code → Unicode script name as used by the `regex` package (\p{Han}, \p{Latin}, ...)
UNICODE_SCRIPT = {
    "Latn": "Latin",
    "Cyrl": "Cyrillic",
    "Hans": "Han",
    "Hant": "Han",
    "Deva": "Devanagari",
    "Arab": "Arabic",
}


def script_of(code: str) -> str:
    """'npi_Deva' → 'Deva'."""
    return code.split("_")[1]


def endonym(code: str) -> str:
    """The language's name in its own language; fails loudly for unregistered languages."""
    if code not in ENDONYMS:
        raise KeyError(f"no endonym registered for {code!r}; add it to latent_switch.languages.ENDONYMS")
    return ENDONYMS[code]


# Languages for the trace experiments (C1), keyed by the short codes used in data/prompts/*.csv.
#   flores : FLORES code (ties into ENDONYMS / script helpers above)
#   mclm   : column name in amphora/MCLM MT-MATH100
#   lcb    : has LCB-Dolly prompts (ne: our own translation of the hi rows)
#   opener : prefill text placed after "<think>\n" in the prefill condition
#            (ne/hi/ru copied from the B1 force test; zh/ja/ar/fr/sw drafted 2026-11-01, unchecked by a native speaker)
#   role   : why it is in the set (resource level x script)
# Japanese (Jpan = Han + Hiragana + Katakana) is deliberately not in UNICODE_SCRIPT: script_mask expects one script.
TRACE_LANGS = {
    "ne": dict(flores="npi_Deva", mclm="ne", lcb=True,
               opener="ठीक छ, नेपालीमा सोचौं। पहिले प्रश्नलाई राम्ररी बुझौं।",
               role="low resource, non-Latin; development language"),
    "hi": dict(flores="hin_Deva", mclm="hi", lcb=True,
               opener="ठीक है, चलिए हिंदी में सोचते हैं। पहले प्रश्न को ध्यान से समझते हैं।",
               role="mid resource, same script as ne"),
    "ru": dict(flores="rus_Cyrl", mclm="ru", lcb=True,
               opener="Хорошо, давайте рассуждать на русском. Сначала внимательно разберём условие задачи.",
               role="high resource, non-Latin; thinks in X (control)"),
    "zh": dict(flores="cmn_Hans", mclm="zh-cn", lcb=False,
               opener="好的，我们用中文来思考。首先仔细理解一下题目。",
               role="Qwen's other dominant language: is English special?"),
    "ja": dict(flores="jpn_Jpan", mclm="ja", lcb=False,
               opener="では、日本語で考えてみましょう。まず問題をよく読んで理解しましょう。",
               role="shares Han with zh: does it drift to Chinese?"),
    "ar": dict(flores="arb_Arab", mclm="ar", lcb=True,
               opener="حسنًا، لنفكر باللغة العربية. أولًا، لنفهم السؤال جيدًا.",
               role="mid resource, other family, RTL script"),
    "fr": dict(flores="fra_Latn", mclm="fr", lcb=True,
               opener="D'accord, réfléchissons en français. D'abord, comprenons bien la question.",
               role="high resource, Latin script: script vs language"),
    "sw": dict(flores="swh_Latn", mclm="sw", lcb=False,
               opener="Sawa, tufikiri kwa Kiswahili. Kwanza, tuelewe swali vizuri.",
               role="low resource, Latin script: resource vs script"),
}
