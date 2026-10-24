"""AJAN 1: Kod Denetçisi (statik analiz, LLM'siz)."""
import ast
from pyflakes.checker import Checker

CLEAN = "1. Ajan: Kod temiz, hata yok."


def review(code: str, filename: str = "kod.py") -> list[str]:
    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError as e:
        return [f"[SÖZDİZİMİ] - [{filename}:{e.lineno}] - {e.msg} -> satırı düzelt"]

    out = []
    for m in sorted(Checker(tree, filename=filename).messages, key=lambda m: m.lineno):
        msg = m.message % m.message_args
        name = type(m).__name__
        fix = {
            "UnusedImport": "import satırını sil",
            "UndefinedName": "tanımla veya import et",
            "UnusedVariable": "değişkeni sil veya kullan",
            "RedefinedWhileUnused": "yinelenen tanımı kaldır",
        }.get(name, "satırı gözden geçir")
        out.append(f"[{name}] - [{filename}:{m.lineno}] - {msg} -> {fix}")

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            out.append(f"[BARE_EXCEPT] - [{filename}:{node.lineno}] - `except Exception as e:` kullan")
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "fit_transform":
            out.append(f"[LEAKAGE_RİSKİ] - [{filename}:{node.lineno}] - fit_transform yalnızca train'e; test'e transform. Pipeline kullan")
    return out or [CLEAN]
