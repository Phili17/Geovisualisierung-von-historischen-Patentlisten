#!/usr/bin/env python3
"""
Texterkennung mit einem Vision-LLM über die Claude-API (Standard: Claude Sonnet 5.5)
auf den mit `bilder_optimieren.py --fuer vlm` erzeugten Spaltenkacheln.

Ausgabe im selben CSV-Format wie text_extrahieren.py (eine Zeile je Textzeile),
damit sich Tesseract und Vision-LLM direkt vergleichen lassen.

Ablauf:
  1. Kachelliste aus <quelle>/<Jahr>/bilder_<Jahr>.csv lesen (Kopf wird übersprungen)
  2. Kosten vorab schätzen; Abbruch, wenn --max-kosten überschritten würde
  3. Jede Kachel einzeln an die API schicken: buchstabengetreue Abschrift,
     Zeile für Zeile wie gedruckt. Antworten werden je Kachel zwischengespeichert
     (JSON mit Text und Token-Verbrauch) -> ein abgebrochener Lauf macht dort weiter.
  4. Kacheln einer Spalte in Reihenfolge zusammensetzen, CSV schreiben:
       <ziel>/<Jahr>/zeilen_<Jahr>_<Modell>.csv

Einstellungen (Claude-Doku, Stand Nov. 2026):
  - Modell claude-sonnet-5-5: $2 / $10 je Mio. Eingabe-/Ausgabe-Tokens
  - temperature/top_p sind bei Sonnet 5.5 nicht erlaubt (400-Fehler) -> nicht gesetzt
  - effort "low" und thinking {"type": "between_tools"} (kein Vorab-Denken):
    für eine reine Abschrift genug und deutlich günstiger; mit --denken einschalten
  - Bilder: Kacheln von bilder_optimieren.py halten die Grenzen ein (2576 px, 4784 Tokens)

Voraussetzungen:
  pip install anthropic
  API-Schlüssel als Umgebungsvariable:  export ANTHROPIC_API_KEY="sk-ant-…"
  (Schlüssel nie ins Skript schreiben oder mit dem Code weitergeben.)

Benutzung:
  python bilder_optimieren.py --jahr 1919 --fuer vlm --probe 3   # Kacheln erzeugen
  python vlm_extrahieren.py --jahr 1919 --nur-schaetzen           # nur Kosten schätzen
  python vlm_extrahieren.py --jahr 1919 --probe 4                 # 4 Kacheln testen
  python vlm_extrahieren.py --jahr 1919 --max-kosten 5            # alles, höchstens 5 $
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Preise in US-$ je Million Tokens (Eingabe, Ausgabe) – bei Bedarf anpassen
PREISE = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-5-5": (0.10, 0.50),     # "ab"-Preis laut Doku
}
# grobe Annahme für die Schätzung: Ausgabe-Tokens je Kachel (Abschrift ~80–110 Zeilen)
AUSGABE_JE_KACHEL = 1400

SYSTEM = (
    "Du bist ein sorgfältiger Transkriptor historischer Drucke. Du schreibst Text "
    "buchstabengetreu ab und erfindest oder ergänzt nichts."
)
AUFGABE = """Das Bild zeigt einen Ausschnitt einer Spalte aus dem „Deutschen Reichsanzeiger“ (1900–1923), gesetzt in Fraktur: Listen von Patenten und Gebrauchsmustern.

Schreibe den Text buchstabengetreu ab:
- Genau eine Ausgabezeile je gedruckter Zeile, in derselben Reihenfolge. Keine Zeilen zusammenfassen oder auslassen.
- Langes s (ſ) als normales s schreiben. Umlaute, Satzzeichen, Abkürzungen, Zahlen und Leerzeichen in Zahlen (z. B. „22 447“) wie gedruckt.
- Trennstriche am Zeilenende (⸗ oder =) als „-“ wiedergeben und das Wort NICHT zusammenziehen.
- Fett gedruckte Klassenbezeichnungen am Eintragsanfang (z. B. „12e,“ oder „45k.“) genau abschreiben.
- Nichts korrigieren, modernisieren oder ergänzen. Unsichere Zeichen trotzdem nach bestem Lesen schreiben; völlig Unlesbares als [?].
- Abgeschnittene Zeilen am oberen oder unteren Bildrand so weit abschreiben, wie sie lesbar sind.

Antworte ausschließlich mit der Abschrift, ohne Einleitung, Kommentar oder Formatierung."""


# ---------------------------------------------------------------- Hilfen

def kachel_infos(pfad):
    """…/19190106_nr004/seite_013_sp2_t1.jpg -> Datum, Ausgabe, Seite, Bereich, Teil."""
    m_o = re.match(r"(\d{8})_nr(\d+)", pfad.parent.name)
    m_d = re.match(r"seite_(\d+)_(.+?)(?:_s(\d+))?_t(\d+)$", pfad.stem)
    datum = m_o.group(1) if m_o else ""
    return {
        "datum": datum,
        "wochentag": time.strftime("%a", time.strptime(datum, "%Y%m%d")) if datum else "",
        "ausgabe": int(m_o.group(2)) if m_o else "",
        "seite": int(m_d.group(1)) if m_d else "",
        "bereich": m_d.group(2) if m_d else pfad.stem,
        "streifen": int(m_d.group(3) or 1) if m_d else 1,
        "teil": int(m_d.group(4)) if m_d else 1,
    }


def kacheln_finden(quelle_jahr, jahr):
    liste = quelle_jahr / f"bilder_{jahr}.csv"
    if liste.exists():
        with open(liste, encoding="utf-8") as f:
            zeilen = list(csv.DictReader(f))
        pfade, tokens = [], {}
        for z in zeilen:
            p = Path(z["kachel"])
            if not p.exists():
                p = quelle_jahr / p.parent.name / p.name
            pfade.append(p)
            tokens[p] = int(z.get("tokens") or 0)
        return pfade, tokens
    pfade = sorted(quelle_jahr.glob("*/seite_*_t*.jpg"))
    return pfade, {}


def kosten(modell, ein, aus):
    p_ein, p_aus = PREISE.get(modell, PREISE["claude-sonnet-5-5"])
    return ein / 1e6 * p_ein + aus / 1e6 * p_aus


# ---------------------------------------------------------------- API

def anfrage(client, pfad, args):
    """Eine Kachel transkribieren. Gibt dict mit text, Tokens, stop_reason zurück."""
    daten = base64.standard_b64encode(pfad.read_bytes()).decode("ascii")
    typ = "image/png" if pfad.suffix.lower() == ".png" else "image/jpeg"
    zusatz = {"output_config": {"effort": args.effort}}
    if not args.denken:
        zusatz["thinking"] = {"type": "between_tools"}      # kein Vorab-Denken
    antwort = client.messages.create(
        model=args.modell,
        max_tokens=args.max_tokens,
        system=SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": typ, "data": daten}},
            {"type": "text", "text": AUFGABE},
        ]}],
        extra_body=zusatz,          # über extra_body: funktioniert auch mit älteren SDK-Versionen
    )
    text = "".join(b.text for b in antwort.content if getattr(b, "type", "") == "text")
    return {"text": text, "stop_reason": antwort.stop_reason, "modell": antwort.model,
            "input_tokens": antwort.usage.input_tokens, "output_tokens": antwort.usage.output_tokens}


def kachel_bearbeiten(client, pfad, cache, args):
    erg = anfrage(client, pfad, args)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    tmp.write_text(json.dumps(erg, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(cache)
    return erg


# ---------------------------------------------------------------- Ablauf

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jahr", required=True)
    ap.add_argument("--quelle", default="farbscans_vlm", help="Ausgabeordner von bilder_optimieren.py --fuer vlm")
    ap.add_argument("--ziel", default="ocr_ergebnisse")
    ap.add_argument("--modell", default="claude-sonnet-5-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                    help="Aufwand des Modells (Standard low; höher = teurer)")
    ap.add_argument("--denken", action="store_true", help="Vorab-Denken erlauben (teurer, ggf. genauer)")
    ap.add_argument("--max-tokens", type=int, default=4000, help="max. Ausgabe-Tokens je Kachel")
    ap.add_argument("--parallel", type=int, default=3, help="gleichzeitige Anfragen (Rate-Limits beachten)")
    ap.add_argument("--max-kosten", type=float, default=2.0, help="Abbruch, wenn die Schätzung höher liegt (US-$)")
    ap.add_argument("--mit-kopf", action="store_true", help="auch Kopfkacheln transkribieren")
    ap.add_argument("--probe", type=int, help="nur die ersten N Kacheln")
    ap.add_argument("--nur-schaetzen", action="store_true", help="nur Kosten schätzen, nichts senden")
    ap.add_argument("--neu", action="store_true", help="Zwischenspeicher ignorieren")
    args = ap.parse_args()

    quelle_jahr = Path(args.quelle) / str(args.jahr)
    kacheln, bild_tokens = kacheln_finden(quelle_jahr, args.jahr)
    if not args.mit_kopf:
        kacheln = [k for k in kacheln if kachel_infos(k)["bereich"] != "kopf"]
    if args.probe:
        kacheln = kacheln[:args.probe]
    if not kacheln:
        raise SystemExit(f"Keine Kacheln unter {quelle_jahr} – erst: "
                         f"python bilder_optimieren.py --jahr {args.jahr} --fuer vlm")

    ziel_jahr = Path(args.ziel) / str(args.jahr)
    cache_ordner = ziel_jahr / "vlm" / args.modell
    def cache_pfad(k):
        return cache_ordner / k.parent.name / (k.stem + ".json")
    offen = [k for k in kacheln if args.neu or not cache_pfad(k).exists()]

    # Kostenschätzung: Bild-Tokens aus der Kachelliste + Prompt + geschätzte Ausgabe
    ein = sum(bild_tokens.get(k, 2000) + 450 for k in offen)
    aus = AUSGABE_JE_KACHEL * len(offen)
    schaetzung = kosten(args.modell, ein, aus)
    print(f"{len(kacheln)} Kacheln, {len(offen)} offen, Modell {args.modell}, effort {args.effort}"
          f"{', mit Denken' if args.denken else ''}")
    print(f"Schätzung: ~{ein:,} Eingabe- + ~{aus:,} Ausgabe-Tokens ≈ {schaetzung:.2f} $"
          f" (Batch-API wäre ~50 % günstiger)")
    if args.nur_schaetzen:
        return
    if offen and schaetzung > args.max_kosten:
        raise SystemExit(f"Schätzung über --max-kosten {args.max_kosten:.2f} $ – Grenze erhöhen oder --probe nutzen")

    fehler = []
    if offen:
        try:
            import anthropic
        except ImportError:
            raise SystemExit("Paket fehlt: pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY ist nicht gesetzt (export ANTHROPIC_API_KEY=…)")
        client = anthropic.Anthropic(max_retries=6)     # wiederholt bei 429/5xx automatisch
        start, verbraucht_ein, verbraucht_aus = time.time(), 0, 0
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            auftraege = {pool.submit(kachel_bearbeiten, client, k, cache_pfad(k), args): k for k in offen}
            for i, a in enumerate(as_completed(auftraege), start=1):
                k = auftraege[a]
                try:
                    erg = a.result()
                    verbraucht_ein += erg["input_tokens"]
                    verbraucht_aus += erg["output_tokens"]
                    warnung = "  ! abgeschnitten (max_tokens)" if erg["stop_reason"] == "max_tokens" else ""
                    print(f"  [{i}/{len(offen)}] {k.parent.name}/{k.name}: "
                          f"{erg['text'].count(chr(10)) + 1} Zeilen, {erg['input_tokens']:,}/{erg['output_tokens']:,} Tokens"
                          f"{warnung}")
                except Exception as e:
                    fehler.append(k)
                    print(f"  ! {k.parent.name}/{k.name}: {type(e).__name__}: {str(e)[:200]}")
        print(f"Verbraucht: {verbraucht_ein:,} Eingabe- / {verbraucht_aus:,} Ausgabe-Tokens "
              f"≈ {kosten(args.modell, verbraucht_ein, verbraucht_aus):.2f} $ "
              f"in {(time.time() - start) / 60:.1f} min")

    # Kacheln je Spalte in Reihenfolge zusammensetzen -> CSV
    csv_pfad = ziel_jahr / f"zeilen_{args.jahr}_{args.modell}.csv"
    felder = ["datum", "wochentag", "ausgabe", "seite", "bereich", "zeile", "text", "text_norm",
              "konfidenz", "x", "y", "breite", "hoehe", "bild", "modell"]
    reihenfolge = sorted(kacheln, key=lambda k: (k.parent.name, kachel_infos(k)["seite"],
                                                 kachel_infos(k)["bereich"], kachel_infos(k)["streifen"],
                                                 kachel_infos(k)["teil"]))
    n, zaehler = 0, {}
    with open(csv_pfad, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=felder)
        w.writeheader()
        for k in reihenfolge:
            cache = cache_pfad(k)
            if not cache.exists():
                continue
            erg = json.loads(cache.read_text(encoding="utf-8"))
            info = kachel_infos(k)
            schluessel = (k.parent.name, info["seite"], info["bereich"])
            for zeile in erg["text"].splitlines():
                if not zeile.strip():
                    continue
                zaehler[schluessel] = zaehler.get(schluessel, 0) + 1
                w.writerow({"datum": info["datum"], "wochentag": info["wochentag"],
                            "ausgabe": info["ausgabe"], "seite": info["seite"], "bereich": info["bereich"],
                            "zeile": zaehler[schluessel], "text": zeile.strip(),
                            "text_norm": zeile.strip().replace("ſ", "s").replace("⸗", "-").replace("=", "-"),
                            "konfidenz": "", "x": "", "y": "", "breite": "", "hoehe": "",
                            "bild": f"{k.parent.name}/{k.name}", "modell": erg.get("modell", args.modell)})
                n += 1
    print(f"\n{n} Textzeilen -> {csv_pfad}")
    if fehler:
        print(f"! {len(fehler)} Kacheln fehlgeschlagen – beim nächsten Start wird es erneut versucht")
        sys.exit(1)


if __name__ == "__main__":
    main()
