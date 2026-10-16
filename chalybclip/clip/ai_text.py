"""Spanish for the AI's own sentences (decision reasons, verdict reasons /
warnings / recommendation, director lines, score explanations, timeline
labels, integrity labels).

The scoring code keeps producing English on purpose: other code matches on
those strings (`director_lines` looks for "No hook…", "UI", "Strong hook…"),
and the dashboard + tests read them. The hub API translates at its output
boundary with `localize(text, locale)`: Spanish unless the caller asked for
English. Each entry mirrors one template in clip/ai_decisions.py,
clip/publishability.py, clip/scoring.py, clip/intelligence.py,
clip/breakdown.py or api/routers/dashboard.py — add the Spanish here when
you add a sentence there. Unknown text passes through unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_EXACT: dict[str, str] = {
    # ai_decisions.py
    "Action/gameplay leads — keeping the bottom HUD clear": (
        "Manda la acción del juego: dejamos libre el HUD de abajo"
    ),
    "Strong reaction · captions tuned down to let the face speak": (
        "Reacción fuerte: bajamos los subtítulos para que hable tu cara"
    ),
    "Voice-triggered clip — captions surface the phrase": (
        "Clip activado por voz: los subtítulos muestran la frase"
    ),
    "Mostly silent — captions skipped": "Casi sin voz: sin subtítulos",
    "Mostly silent clip — captions disabled": "Clip casi sin voz: subtítulos apagados",
    "Some speech detected — captions on (standard)": (
        "Se detectó algo de voz: subtítulos encendidos (normal)"
    ),
    "Clip is too short — hook would be missed": "El clip es muy corto: nadie leería el hook",
    "Pure reaction · hook would crowd the moment": (
        "Pura reacción: un hook le quitaría espacio al momento"
    ),
    "Pure reaction · main hook hidden so face leads": (
        "Pura reacción: escondimos el hook para que mande tu cara"
    ),
    "Hook recommended · sets up the spoken trigger": (
        "Conviene un hook: prepara la frase que dijiste"
    ),
    "Hook recommended · primes the reaction payoff": ("Conviene un hook: prepara la reacción"),
    "Hook recommended · standard short-form template": (
        "Conviene un hook: la fórmula de siempre en video corto"
    ),
    "Long voice-trigger clip · context line helps setup": (
        "Clip largo activado por voz: una línea de contexto ayuda"
    ),
    "Context line recommended above the main hook": (
        "Conviene una línea de contexto arriba del hook"
    ),
    "Clip too short for two hooks": "El clip es muy corto para dos hooks",
    "Single hook reads cleaner here": "Aquí se lee mejor un solo hook",
    "Face high in frame → captions below it": "Tu cara va arriba → subtítulos debajo",
    "Face low in frame → captions above it": "Tu cara va abajo → subtítulos encima",
    "Gameplay HUD likely in bottom band → captions up high": (
        "El HUD del juego suele ir abajo → subtítulos arriba"
    ),
    "Kick has no IG-style dock → captions can ride low": (
        "Kick no tiene barra abajo → los subtítulos pueden ir bajos"
    ),
    "Standard short-form placement (under the face)": (
        "Posición normal en video corto (debajo de tu cara)"
    ),
    # publishability.py
    "Strong hook — viewers see the payoff in the first second": (
        "Hook fuerte: se ve lo bueno desde el primer segundo"
    ),
    "Hook text present": "Tiene texto de hook",
    "Captions are at a comfortable read pace": "Los subtítulos van a un ritmo cómodo de leer",
    "Strong face presence — viewers see the subject clearly": (
        "Tu cara se ve mucho: se te reconoce claro"
    ),
    "Subject visible most of the clip": "Sales en casi todo el clip",
    "Motion / visual interest is high": "Hay mucho movimiento e interés visual",
    "Fix the highlighted issues — they will be visible on the published post": (
        "Corrige lo marcado: se va a notar en la publicación"
    ),
    "Add a hook headline up top — every viral clip starts with one": (
        "Ponle un título de hook arriba: todo clip viral empieza con uno"
    ),
    "Turn captions on — they're the #1 retention driver on muted scrolls": (
        "Prende los subtítulos: es lo que más retiene cuando ven sin sonido"
    ),
    "Click Apply AI fixes — most warnings auto-resolve": (
        "La IA ya corrigió lo que pudo; revisa lo que queda"
    ),
    "Ready to ship — Publish Confidence is high": "Listo para publicar: la confianza es alta",
    "Review the listed warnings, then re-check confidence": (
        "Revisa los avisos y vuelve a checar la confianza"
    ),
    "Strong face presence detected": "Se detectó tu cara con claridad",
    "Vision LLM rated this a strong reaction moment": (
        "La IA de imagen lo calificó como una reacción fuerte"
    ),
    "Motion energy is high — visually punchy": "Mucho movimiento: se ve con fuerza",
    "Long silent stretches — dead-air risk is high": "Tramos largos sin voz: riesgo alto de silencios",
    "Speech is too fast for captions to keep up": "Hablas tan rápido que los subtítulos no alcanzan",
    "Hook is sharp and clear": "El hook es claro y directo",
    "Hook is missing — first second is silent": "Falta el hook: el primer segundo no dice nada",
    "stream integrity issue": "problema en el video del stream",
    # scoring.py
    "no transcript yet — can't measure caption pacing": (
        "todavía no hay transcripción: no se puede medir el ritmo"
    ),
    "no transcript — can't measure dead-air": "sin transcripción: no se pueden medir los silencios",
    # intelligence.py
    "Scene cut": "Cambio de escena",
}

_EMOTIONS: dict[str, str] = {
    "neutral": "neutral",
    "happy": "feliz",
    "surprise": "sorpresa",
    "surprised": "sorpresa",
    "angry": "enojo",
    "sad": "tristeza",
    "fear": "miedo",
    "disgust": "asco",
}

_HEURISTICS: dict[str, str] = {
    "voice": "tu voz",
    "audio": "un pico de audio",
    "visual": "algo visual",
    "chat": "el chat",
    "chat heat": "el chat",
    "interval": "un intervalo",
}


_P = Callable[[re.Match[str]], str]
_PATTERNS: list[tuple[re.Pattern[str], _P]] = [
    # ai_decisions.py
    (
        re.compile(r"^Face on screen (\d+)% of the clip — face leads$"),
        lambda m: f"Tu cara sale el {m[1]}% del clip: manda tu cara",
    ),
    (
        re.compile(r"^Active speaking \(([\d.]+) wps\) — captions essential$"),
        lambda m: f"Hablas mucho ({m[1]} palabras/s): los subtítulos son clave",
    ),
    (
        re.compile(r"^Clip is too short \(([\d.]+)s\) to read a hook$"),
        lambda m: f"El clip es muy corto ({m[1]} s) para leer un hook",
    ),
    # publishability.py
    (
        re.compile(r"^Captions / banner / hook are safe for (.+) UI$"),
        lambda m: f"Subtítulos, banner y hook no los tapa la pantalla de {m[1]}",
    ),
    (
        re.compile(r"^stream integrity: (\d+) issues detected — re-trim or skip$"),
        lambda m: f"El video del stream tiene {m[1]} problemas: recórtalo o descártalo",
    ),
    # breakdown.py integrity labels
    (
        re.compile(r"^freeze: (\d+)s zero-motion at (\d+)-(\d+)s$"),
        lambda m: f"imagen congelada {m[1]} s (del {m[2]} al {m[3]} s)",
    ),
    (
        re.compile(r"^silence: (\d+)s without speech at (\d+)-(\d+)s$"),
        lambda m: f"silencio de {m[1]} s (del {m[2]} al {m[3]} s)",
    ),
    # scoring.py
    (
        re.compile(r"^viral (\d+) ≥ 60 — push this clip first$"),
        lambda m: f"viral {m[1]} ≥ 60: publica este primero",
    ),
    (
        re.compile(r"^viral (\d+) in 40-59 — solid candidate$"),
        lambda m: f"viral {m[1]} entre 40 y 59: buen candidato",
    ),
    (
        re.compile(r"^viral (\d+) < 40 — a strong hook can lift it$"),
        lambda m: f"viral {m[1]} < 40: un buen hook lo puede subir",
    ),
    (
        re.compile(r"^([\d.]+) words/sec — comfortable read pace$"),
        lambda m: f"{m[1]} palabras/s: ritmo cómodo de leer",
    ),
    (
        re.compile(r"^([\d.]+) words/sec — viewers may struggle$"),
        lambda m: f"{m[1]} palabras/s: puede costar leerlos",
    ),
    (
        re.compile(r"^([\d.]+) words/sec — outside the readable band$"),
        lambda m: f"{m[1]} palabras/s: fuera del ritmo legible",
    ),
    (
        re.compile(r"^([\d.]+) words/sec — clip is talking through$"),
        lambda m: f"{m[1]} palabras/s: hablas todo el clip",
    ),
    (
        re.compile(r"^([\d.]+) words/sec — some quiet stretches$"),
        lambda m: f"{m[1]} palabras/s: hay algunos ratos callados",
    ),
    (
        re.compile(r"^([\d.]+) words/sec — long quiet stretches; trim or skip$"),
        lambda m: f"{m[1]} palabras/s: ratos largos callado; recórtalo o descártalo",
    ),
    # intelligence.py
    (
        re.compile(r"^Audio peak · (\d+) words$"),
        lambda m: f"Pico de audio · {m[1]} palabras",
    ),
    (
        re.compile(r"^Reaction · (\d+) laugh tokens?$"),
        lambda m: f"Reacción · {m[1]} risa{'s' if m[1] != '1' else ''}",
    ),
    (
        re.compile(r"^Motion peak · ([\d.]+)$"),
        lambda m: f"Pico de movimiento · {m[1]}",
    ),
    (
        re.compile(r"^Face emotion → (.+)$"),
        lambda m: f"Emoción en la cara → {_EMOTIONS.get(m[1], m[1])}",
    ),
    (
        re.compile(r"^Chat spike · (\d+) messages$"),
        lambda m: f"Pico de chat · {m[1]} mensajes",
    ),
    (
        re.compile(r'^🎯 (Trigger phrase heard|Voice trigger fired): "(.*)"$', re.S),
        lambda m: f"🎯 La IA te escuchó: «{m[2]}»",
    ),
    (
        re.compile(r'^🎯 (Retroactive trigger heard|Retroactive trigger fired): "(.*)"$', re.S),
        lambda m: f"🎯 La IA te escuchó (hacia atrás): «{m[2]}»",
    ),
    # dashboard.py intel line (face_zone) and heuristic reason
    (re.compile(r"^Detectamos: (.+)$"), lambda m: f"Detectamos: {_HEURISTICS.get(m[1], m[1])}"),
    # platform_zones collision messages: "<Overlay>: <reason>"
    (re.compile(r"^(Banner|Captions|Top_Hook|Top Hook): (.+)$"), lambda m: f"{m[1]}: {m[2]}"),
]

def localize(text: str, locale: str = "es") -> str:
    """`text` in `locale` (es by default). Only `en` keeps the original."""
    if not isinstance(text, str) or locale == "en":
        return text
    hit = _EXACT.get(text)
    if hit is not None:
        return hit
    for pat, fn in _PATTERNS:
        m = pat.match(text)
        if m:
            return fn(m)
    return text


def localize_viral_why(text: str, locale: str = "es") -> str:
    """The viral score's "baseline 55 · reaction 0.62 → +9 · …" trail."""
    if locale == "en" or not isinstance(text, str):
        return text
    out = text.replace("baseline", "base").replace("reaction", "reacción")
    out = out.replace("heuristic", "detector").replace("face", "cara")
    out = out.replace("speaking active", "habla activa").replace("some speech", "algo de voz")
    return out


__all__ = ["localize", "localize_viral_why"]
