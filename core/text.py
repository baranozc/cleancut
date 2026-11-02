import unicodedata


def normalize_word(word: str) -> str:
    """
    Kelimeyi karşılaştırma için sadeleştirir: Türkçe kurallarıyla küçük harfe çevirir
    ve harf/rakam dışındaki her şeyi (noktalama, kesme işareti, tırnak) atar.

    "piçin," → "piçin" | "Bi’" → "bi" | "İşte" → "işte" | "ISRAR" → "ısrar"

    ı/i ayrımı korunur ("sık" ile "sik" aynı kelime sayılmaz).
    """
    word = unicodedata.normalize("NFC", word).replace("I", "ı").replace("İ", "i").lower()
    return "".join(ch for ch in word if ch.isalnum())


def normalize_words(text: str) -> list[str]:
    """Metni normalize edilmiş kelimelere böler; yalnızca noktalamadan oluşan parçaları atar."""
    return [w for w in (normalize_word(t) for t in text.split()) if w]


def find_sequence(haystack: list[str], needle: list[str]) -> list[int]:
    """needle kelime dizisinin haystack içinde başladığı tüm indeksleri döner."""
    n = len(needle)
    if n == 0:
        return []
    return [i for i in range(len(haystack) - n + 1) if haystack[i:i + n] == needle]
