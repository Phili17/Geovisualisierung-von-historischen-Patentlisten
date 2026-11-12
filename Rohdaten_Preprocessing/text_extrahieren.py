#!/usr/bin/env python3
"""
Generiert mit Opus 5.5 mit folgenden Anforderungen:
Texterkennung (Tesseract 5, Fraktur-Modell) auf den mit bilder_optimieren.py
aufbereiteten Spaltenbildern – Ergebnis als CSV mit einer Zeile je Textzeile.

Ablauf:
  1. Bildliste aus <quelle>/<Jahr>/bilder_<Jahr>.csv lesen (von bilder_optimieren.py)
     – fehlt sie, werden alle PNGs unter <quelle>/<Jahr>/ genommen
  2. Jedes Bild mit Tesseract erkennen (TSV-Ausgabe: Wörter mit Position und
     Konfidenz), mehrere Bilder parallel. Das Rohergebnis je Bild wird
     zwischengespeichert -> ein abgebrochener Lauf macht dort weiter.
  3. Wörter zu Zeilen zusammensetzen und alles in eine CSV schreiben:
       <ziel>/<Jahr>/zeilen_<Jahr>_<Modell>.csv

Spalten der CSV:
  datum, wochentag, ausgabe, seite, bereich (kopf, sp1, sp2, …), zeile (Nr. im Bild),
  text (wie erkannt, mit ſ), text_norm (ſ->s, ⸗/= -> -), konfidenz (Mittel der Wörter, 0–100),
  x, y, breite, hoehe (Zeile im Bild, Pixel), bild, modell

Voraussetzungen:
  - Tesseract 5 
  - ein Fraktur-Modell als .traineddata, am besten aus tessdata_best:
      https://github.com/tesseract-ocr/tessdata_best/raw/main/script/Fraktur.traineddata
    in einen Ordner legen (z. B. ./tessdata) und mit --tessdata angeben.
    Andere Modelle (deu_latf, frak2021 der UB Mannheim) über --modell wählen.
    Hinweis: Homebrews "tesseract-lang" enthält nur die schnelleren, ungenaueren
    tessdata_fast-Modelle.

Benutzung:
  python text_extrahieren.py --jahr 1919 --tessdata tessdata
  python text_extrahieren.py --jahr 1919 --tessdata tessdata --modell deu_latf
  python text_extrahieren.py --jahr 1919 --tessdata tessdata --probe 5 --prozesse 4
  Der Seitenkopf (Titel) wird standardmäßig übersprungen; --mit-kopf nimmt ihn dazu.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------- Tesseract

def tesseract_pruefen(args):
    try:
        version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("Tesseract nicht gefunden – installieren (macOS: brew install tesseract)")
    erste = (version.stdout or version.stderr).splitlines()[0]
    m = re.search(r"(\d+)\.(\d+)", erste)
    if not m or int(m.group(1)) < 5:
        raise SystemExit(f"Tesseract 5 benötigt, gefunden: {erste}")
    cmd = ["tesseract", "--list-langs"] + (["--tessdata-dir", args.tessdata] if args.tessdata else [])
    ausgabe = subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()
    modelle = [z.strip() for z in ausgabe[1:] if z.strip()]      # 1. Zeile ist eine Überschrift
    if args.modell not in modelle:
        raise SystemExit(f"Modell '{args.modell}' nicht gefunden"
                         f"{' in ' + args.tessdata if args.tessdata else ''}. "
                         f"Vorhanden: {', '.join(modelle) or '–'}")
    return erste


def tesseract_tsv(bild, modell, psm, tessdata):
    """Tesseract mit TSV-Ausgabe (Wörter mit Position und Konfidenz)."""
    cmd = ["tesseract", str(bild), "stdout", "-l", modell, "--psm", str(psm),
           "-c", "tessedit_create_tsv=1"]
    if tessdata:
        cmd += ["--tessdata-dir", str(tessdata)]
    # ein Thread je Tesseract-Prozess – parallelisiert wird über mehrere Prozesse
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    erg = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if erg.returncode != 0:
        raise RuntimeError(erg.stderr.strip()[:300])
    return erg.stdout


def bild_erkennen(bild, cache, modell, psm, tessdata):
    """Ein Bild erkennen und das TSV im Zwischenspeicher ablegen (läuft im Unterprozess)."""
    start = time.time()
    tsv = tesseract_tsv(bild, modell, psm, tessdata)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    tmp.write_text(tsv, encoding="utf-8")
    tmp.replace(cache)
    return time.time() - start


# ---------------------------------------------------------------- TSV -> Zeilen

def zeilen_aus_tsv(tsv):
    """Wörter (level 5) zu Zeilen gruppieren, in Lesereihenfolge von Tesseract."""
    zeilen = {}
    reihenfolge = []
    for z in csv.DictReader(tsv.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE):
        if z.get("level") != "5" or not (z.get("text") or "").strip():
            continue
        schluessel = (int(z["block_num"]), int(z["par_num"]), int(z["line_num"]))
        if schluessel not in zeilen:
            zeilen[schluessel] = []
            reihenfolge.append(schluessel)
        zeilen[schluessel].append(z)
    ergebnis = []
    for s in reihenfolge:
        woerter = zeilen[s]
        x0 = min(int(w["left"]) for w in woerter)
        y0 = min(int(w["top"]) for w in woerter)
        x1 = max(int(w["left"]) + int(w["width"]) for w in woerter)
        y1 = max(int(w["top"]) + int(w["height"]) for w in woerter)
        konf = [float(w["conf"]) for w in woerter if float(w["conf"]) >= 0]
        ergebnis.append({
            "text": " ".join(w["text"] for w in woerter),
            "konfidenz": round(sum(konf) / len(konf), 1) if konf else "",
            "x": x0, "y": y0, "breite": x1 - x0, "hoehe": y1 - y0,
        })
    return ergebnis


def normalisieren(text):
    return text.replace("ſ", "s").replace("⸗", "-").replace("=", "-")


def bild_infos(pfad):
    """Datum, Ausgabe, Seite, Bereich aus Ordner-/Dateinamen
    (…/19190505_nr101/seite_029_sp2.png)."""
    m_ordner = re.match(r"(\d{8})_nr(\d+)", pfad.parent.name)
    m_datei = re.match(r"seite_(\d+)_(.+)$", pfad.stem)
    datum = m_ordner.group(1) if m_ordner else ""
    wochentag = time.strftime("%a", time.strptime(datum, "%Y%m%d")) if datum else ""
    return {"datum": datum, "wochentag": wochentag,
            "ausgabe": int(m_ordner.group(2)) if m_ordner else "",
            "seite": int(m_datei.group(1)) if m_datei else "",
            "bereich": m_datei.group(2) if m_datei else pfad.stem}


# ---------------------------------------------------------------- Ablauf

def bilder_finden(quelle_jahr, jahr):
    liste = quelle_jahr / f"bilder_{jahr}.csv"
    if liste.exists():
        with open(liste, encoding="utf-8") as f:
            pfade = [Path(z["kachel"]) for z in csv.DictReader(f)]
        # Pfade in der Liste sind relativ zum Startordner von bilder_optimieren.py
        pfade = [p if p.exists() else quelle_jahr / p.parent.name / p.name for p in pfade]
    else:
        pfade = sorted(quelle_jahr.glob("*/seite_*.png"))
    return [p for p in pfade if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jahr", required=True)
    ap.add_argument("--quelle", default="farbscans_tesseract", help="Ausgabeordner von bilder_optimieren.py")
    ap.add_argument("--ziel", default="ocr_ergebnisse")
    ap.add_argument("--modell", default="Fraktur", help="Tesseract-Modell (Standard: Fraktur)")
    ap.add_argument("--tessdata", help="Ordner mit den .traineddata-Dateien")
    ap.add_argument("--psm", type=int, default=4, help="Seitensegmentierung für Spalten (4 = eine Spalte)")
    ap.add_argument("--mit-kopf", action="store_true",
                    help="auch den Seitenkopf erkennen (Titel, Datum; große Schriften -> viele Fehler)")
    ap.add_argument("--psm-kopf", type=int, default=3, help="Seitensegmentierung für den Kopf (3 = automatisch)")
    ap.add_argument("--prozesse", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallele Tesseract-Prozesse (Standard: Kerne – 1)")
    ap.add_argument("--probe", type=int, help="nur die ersten N Bilder (zum Testen)")
    ap.add_argument("--neu", action="store_true", help="Zwischenspeicher ignorieren, alles neu erkennen")
    args = ap.parse_args()

    version = tesseract_pruefen(args)
    quelle_jahr = Path(args.quelle) / str(args.jahr)
    bilder = bilder_finden(quelle_jahr, args.jahr)
    if not args.mit_kopf:
        bilder = [b for b in bilder if not b.stem.endswith("_kopf")]
    if args.probe:
        bilder = bilder[:args.probe]
    if not bilder:
        raise SystemExit(f"Keine Bilder unter {quelle_jahr} gefunden – erst bilder_optimieren.py ausführen")

    ziel_jahr = Path(args.ziel) / str(args.jahr)
    cache_ordner = ziel_jahr / "tsv" / args.modell
    def cache_pfad(b):
        return cache_ordner / b.parent.name / (b.stem + ".tsv")

    offen = [b for b in bilder if args.neu or not cache_pfad(b).exists()]
    print(f"{version}, Modell {args.modell}, {len(bilder)} Bilder "
          f"({len(bilder) - len(offen)} schon erkannt), {args.prozesse} parallele Prozesse")

    fehler = []
    if offen:
        start = time.time()
        with ProcessPoolExecutor(max_workers=args.prozesse) as pool:
            auftraege = {pool.submit(bild_erkennen, b, cache_pfad(b), args.modell,
                                     args.psm_kopf if b.stem.endswith("_kopf") else args.psm,
                                     args.tessdata): b for b in offen}
            for i, auftrag in enumerate(as_completed(auftraege), start=1):
                b = auftraege[auftrag]
                try:
                    dauer = auftrag.result()
                    vergangen = time.time() - start
                    rest = vergangen / i * (len(offen) - i)
                    print(f"  [{i}/{len(offen)}] {b.parent.name}/{b.name}: {dauer:.0f} s "
                          f"(noch ca. {rest / 60:.0f} min)")
                except Exception as e:
                    fehler.append(b)
                    print(f"  ! {b.parent.name}/{b.name}: {e}")

    # alle Zwischenergebnisse in Bild-Reihenfolge zu einer CSV zusammenführen
    csv_pfad = ziel_jahr / f"zeilen_{args.jahr}_{args.modell}.csv"
    felder = ["datum", "wochentag", "ausgabe", "seite", "bereich", "zeile", "text", "text_norm",
              "konfidenz", "x", "y", "breite", "hoehe", "bild", "modell"]
    n_zeilen, konf_summe, konf_n = 0, 0.0, 0
    with open(csv_pfad, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=felder)
        w.writeheader()
        for b in bilder:
            cache = cache_pfad(b)
            if not cache.exists():
                continue
            infos = bild_infos(b)
            for nr, z in enumerate(zeilen_aus_tsv(cache.read_text(encoding="utf-8")), start=1):
                w.writerow({**infos, "zeile": nr, **z, "text_norm": normalisieren(z["text"]),
                            "bild": f"{b.parent.name}/{b.name}", "modell": args.modell})
                n_zeilen += 1
                if z["konfidenz"] != "":
                    konf_summe += z["konfidenz"]
                    konf_n += 1

    print(f"\n{n_zeilen} Textzeilen aus {len(bilder) - len(fehler)} Bildern -> {csv_pfad}")
    if konf_n:
        print(f"Mittlere Konfidenz: {konf_summe / konf_n:.1f} (Tesseract-Schätzung, keine gemessene Genauigkeit)")
    if fehler:
        print(f"! {len(fehler)} Bilder fehlgeschlagen – beim nächsten Start wird es erneut versucht")
        sys.exit(1)


if __name__ == "__main__":
    main()
