import re

def clean_lyrics(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # hapus penanda struktur lagu: [Verse], [Chorus], [Bridge], dll.
    text = re.sub(r"\[.*?\]", " ", text)

    # normalisasi newline → spasi
    text = re.sub(r"\n+", " ", text)

    # normalisasi spasi berlebih
    text = re.sub(r"\s+", " ", text)

    return text.strip()

