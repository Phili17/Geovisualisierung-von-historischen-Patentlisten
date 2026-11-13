#!/usr/bin/env python3
"""
Generiert mit Opus 5.5 nach Dialog/Abwägung über Bildparameter. 

Farbscans der Patent-/Gebrauchsmusterlisten für die Texterkennung aufbereiten:
standardmäßig für Tesseract (Fraktur-Modell), wahlweise für ein Vision-LLM.

Ablauf je Seite:
  1. Graustufen, Schräglage korrigieren
  2. Hintergrund ausgleichen (vergilbtes/fleckiges Papier -> gleichmäßig weiß),
     Kontrast strecken – keine Binarisierung (das macht Tesseract selbst; auf dem
     ausgeglichenen Bild gelingt seine globale Schwelle deutlich besser)
  3. Spalten erkennen (wie spalten_ocr.py), seitenbreiten Kopf getrennt
  4a. --fuer tesseract (Standard): jede Spalte als EIN Bild in Originalgröße
      (Skalierung 1.0), Reste der Spaltenlinien entfernt, Papier ab Grauwert 200
      auf reines Weiß gesetzt, 20 px weißer Rand, verlustfrei als PNG gespeichert
  4b. --fuer vlm: Spalten verkleinert (0.6) und in Kacheln innerhalb der Modellgrenzen
      geteilt (Claude 4.7+/Sonnet 5.5: lange Kante 2576 px, max. 4784 Bild-Tokens),
      Graustufen-JPEG (Qualität 90)
  5. Kontrollbild je Seite + Übersicht (CSV)

Warum für Tesseract Skalierung 1.0 und PNG? Test auf 19 Goldstandard-Zeilen
(5.5.1919, S. 29, Spalte 2, Modell „Fraktur“, Zeichenfehlerrate):
  Original-Graustufen ohne Ausgleich     12,5 %
  Ausgleich, Skalierung 1.0, PNG          6,5–6,9 %
  dto., Papier > 200 -> weiß, PNG         6,8 %   <- Standard (Datei 57 % kleiner)
  dto., nur 16 Graustufen, PNG            8,3 %
  Ausgleich, Skalierung 1.0, JPEG q95     7,2 %   (q90: 11,4 %)
  Ausgleich, Skalierung 0.6              16,0 %   (x-Höhe nur ~10 px)
  Ausgleich, Skalierung 1.5              14,6 %
  (kleine Stichprobe – Richtwert, keine belastbare Messung)

Benutzung:
  pip install opencv-python-headless numpy
  python bilder_optimieren.py --jahr 1919 --probe 3        # erst 3 Seiten testen
  python bilder_optimieren.py --jahr 1919                  # alle Seiten, für Tesseract
  python bilder_optimieren.py --jahr 1919 --fuer vlm       # für ein Vision-LLM (z. B. Sonnet 5.5)

Eingabe:  <quelle>/<Jahr>/<Ausgabe>/seite_NNN.jpg   (Standard: farbscans_treffer)
Ausgabe:  tesseract: farbscans_tesseract/<Jahr>/<Ausgabe>/seite_NNN_kopf.png, seite_NNN_sp1.png, …
          vlm:       farbscans_vlm/<Jahr>/<Ausgabe>/seite_NNN_sp1_t1.jpg, …
          dazu <ziel>/<Jahr>/bilder_<Jahr>.csv und <ziel>/<Jahr>/kontrolle/
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------- Bildaufbereitung

def schraeglage(grau, max_grad=2.0, schritt=0.1):
    """Winkel, bei dem das waagrechte Projektionsprofil am schärfsten ist."""
    _, bw = cv2.threshold(cv2.GaussianBlur(grau, (3, 3), 0), 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    klein = cv2.resize(bw, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    h, w = klein.shape
    bester, beste_var = 0.0, -1
    for winkel in np.arange(-max_grad, max_grad + schritt, schritt):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), winkel, 1.0)
        rot = cv2.warpAffine(klein, m, (w, h), flags=cv2.INTER_NEAREST)
        var = np.var(rot.sum(axis=1).astype(np.float64))
        if var > beste_var:
            bester, beste_var = winkel, var
    return float(bester)


def drehen(bild, winkel, rand):
    h, w = bild.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), winkel, 1.0)
    return cv2.warpAffine(bild, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=rand)


def hintergrund_ausgleichen(grau):
    """Papierfarbe glätten: durch geschätzten Hintergrund teilen (große Schließung
    entfernt die Schrift aus der Schätzung), dann Kontrast strecken.
    Ergebnis: weißes Papier, dunkle Schrift, Graustufen bleiben erhalten."""
    klein = cv2.resize(grau, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    hg = cv2.morphologyEx(klein, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    hg = cv2.medianBlur(hg, 21)
    hg = cv2.resize(hg, (grau.shape[1], grau.shape[0]), interpolation=cv2.INTER_LINEAR)
    norm = cv2.divide(grau, np.maximum(hg, 1), scale=255)
    lo = np.percentile(norm, 1)                    # dunkelste Schrift -> schwarz
    hi = np.percentile(norm, 60)                   # Papier -> weiß
    gestreckt = np.clip((norm.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255)
    return gestreckt.astype(np.uint8)


def binarisieren(grau):
    """Nur für die Layoutanalyse (Text = 255)."""
    grau = cv2.GaussianBlur(grau, (3, 3), 0)
    bw = cv2.adaptiveThreshold(grau, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 15)
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def linien_entfernen(bild):
    """Senkrechte Linienreste (Spaltenlinien am Rand des Zuschnitts) weiß übermalen –
    Tesseract liest sie sonst als „|“ oder „l“."""
    bw = binarisieren(bild)
    linien = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45)))
    linien = cv2.dilate(linien, np.ones((3, 5), np.uint8))
    sauber = bild.copy()
    sauber[linien > 0] = 255
    return sauber


def weisser_rand(bild, rand=20):
    """Tesseract erkennt Text am Bildrand schlechter – etwas Rand hinzufügen."""
    return cv2.copyMakeBorder(bild, rand, rand, rand, rand, cv2.BORDER_CONSTANT, value=255)


# ---------------------------------------------------------------- Layout (aus spalten_ocr.py)

def glaetten(profil, fenster):
    return np.convolve(profil, np.ones(fenster), mode="same")


def oeffnen_1d(profil, breite):
    """1D-Öffnung: entfernt Spitzen schmaler als `breite` (z. B. dünne Spaltenlinien)."""
    from numpy.lib.stride_tricks import sliding_window_view
    r = breite // 2
    p = np.pad(profil, r, mode="edge")
    mn = sliding_window_view(p, 2 * r + 1).min(axis=1)
    p = np.pad(mn, r, mode="edge")
    return sliding_window_view(p, 2 * r + 1).max(axis=1)


def textdichte(bw, linienbreite=14):
    h = bw.shape[0]
    senkrecht = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    profil = cv2.subtract(bw, senkrecht).sum(axis=0) / 255 / max(h, 1)
    return glaetten(oeffnen_1d(profil, linienbreite), 5) / 5


def grenzen_fuer_n(profil, links, rechts, n):
    breite = (rechts - links) / n
    grenzen = [links]
    for i in range(1, n):
        mitte = links + i * breite
        a, b = int(mitte - 0.25 * breite), int(mitte + 0.25 * breite)
        grenzen.append(a + int(np.argmin(profil[a:b])))
    grenzen.append(rechts)
    return grenzen


def baender(hoehe, band_hoehe, bw=None, suchradius=60):
    n = max(1, round(hoehe / band_hoehe))
    kanten = np.linspace(0, hoehe, n + 1).astype(int)
    if bw is not None:
        zeilen = glaetten(bw.sum(axis=1) / 255, 5)
        for i in range(1, n):
            a, b = max(0, kanten[i] - suchradius), min(hoehe, kanten[i] + suchradius)
            kanten[i] = a + int(np.argmin(zeilen[a:b]))
    return list(zip(kanten[:-1], kanten[1:]))


def spaltengrenzen(bw, min_spaltenbreite, n_vorgabe=None, max_spalten=8, band_hoehe=500):
    """[(y_a, y_b, [x0, …, xn]), …] je waagrechtem Band; folgt gebogenen Spalten."""
    h, w = bw.shape
    profil = textdichte(bw)
    belegt = np.where(oeffnen_1d(profil, 80) > 0.3 * np.median(profil[profil > 0]))[0]
    links, rechts = (int(belegt[0]), int(belegt[-1])) if len(belegt) else (0, w - 1)
    band_profile = [(ya, yb, textdichte(bw[ya:yb])) for ya, yb in baender(h, band_hoehe, bw)]

    def bewerten(grob):
        radius = int(0.12 * (rechts - links) / (len(grob) - 1))
        ergebnis, gut = [], 0
        for ya, yb, bp in band_profile:
            med = float(np.median(bp[links:rechts])) or 1e-9
            xs, ok = [links], True
            for x in grob[1:-1]:
                a, b = max(0, x - radius), min(len(bp), x + radius)
                xm = a + int(np.argmin(bp[a:b]))
                xs.append(xm)
                ok &= bp[xm] < 0.10 * med
            xs.append(rechts)
            ergebnis.append((ya, yb, xs))
            gut += ok
        return ergebnis, gut / len(band_profile)

    if n_vorgabe:
        return bewerten(grenzen_fuer_n(profil, links, rechts, n_vorgabe))[0]
    beste = [(0, h, [links, rechts])]
    for n in range(2, max_spalten + 1):
        if (rechts - links) / n < min_spaltenbreite:
            break
        ergebnis, anteil = bewerten(grenzen_fuer_n(profil, links, rechts, n))
        if anteil >= 0.7:
            beste = ergebnis
    return beste


def koerper_start(bw, xs, max_anteil=0.35, fenster=50, folge=4, spiel=60):
    """Beginn des Spaltenteils (nach der letzten Zeile, die eine Spaltenlücke kreuzt)."""
    if len(xs) <= 2:
        return 0
    h = bw.shape[0]
    senkrecht = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    n_fenster = int(h * max_anteil) // fenster
    sauber = np.ones(n_fenster, dtype=bool)
    for x in xs[1:-1]:
        streifen = bw[:int(h * max_anteil), max(0, x - spiel):x + spiel]
        streifen = cv2.subtract(streifen, cv2.morphologyEx(streifen, cv2.MORPH_OPEN, senkrecht))
        for i in range(n_fenster):
            tinte = streifen[i * fenster:(i + 1) * fenster].sum(axis=0) / 255
            sauber[i] &= np.convolve(tinte, np.ones(10), mode="valid").min() <= 3
    schmutzig = np.where(~sauber)[0]
    if not len(schmutzig):
        return 0
    i = int(schmutzig[-1]) + 1
    return i * fenster if sauber[i:i + folge].all() else 0


def textbereich_oben(bw):
    """Erste Zeile mit nennenswert Text (oberer Rand wird abgeschnitten)."""
    zeilen = np.where(bw.sum(axis=1) / 255 > bw.shape[1] * 0.005)[0]
    return max(0, int(zeilen[0]) - 10) if len(zeilen) else 0


def textbereich_unten(bw):
    zeilen = np.where(bw.sum(axis=1) / 255 > bw.shape[1] * 0.005)[0]
    return min(bw.shape[0], int(zeilen[-1]) + 10) if len(zeilen) else bw.shape[0]


# ---------------------------------------------------------------- Kacheln

def tokens(b, h):
    """Bild-Tokens laut Claude-Doku: ein Token je 28x28-Pixel-Feld."""
    return math.ceil(b / 28) * math.ceil(h / 28)


def max_kachelhoehe(breite, max_kante, max_tokens):
    """Größte Kachelhöhe, bei der lange Kante und Token-Grenze eingehalten sind."""
    nach_tokens = (max_tokens // math.ceil(breite / 28)) * 28
    return max(28, min(max_kante, nach_tokens))


def in_kacheln_teilen(bild, max_kante, max_tokens, suchradius=40):
    """Senkrechter Streifen -> Kacheln; Schnitte in den hellsten Zeilen
    (Zeilenzwischenraum) nahe der Höchsthöhe, damit keine Textzeile zerschnitten wird."""
    h, b = bild.shape
    grenze = max_kachelhoehe(b, max_kante, max_tokens)
    if h <= grenze:
        return [bild]
    tinte = glaetten((255 - bild).sum(axis=1).astype(np.float64), 3)
    kacheln, y = [], 0
    while h - y > grenze:
        ziel = y + grenze
        a = max(y + grenze // 2, ziel - 2 * suchradius)
        schnitt = a + int(np.argmin(tinte[a:ziel]))     # nie über die Grenze hinaus
        kacheln.append(bild[y:schnitt])
        y = schnitt
    kacheln.append(bild[y:])
    return kacheln


def skalieren(bild, faktor):
    if abs(faktor - 1) < 1e-3:
        return bild
    return cv2.resize(bild, None, fx=faktor, fy=faktor, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- Seite

def seite_verarbeiten(pfad, ziel, kontroll_ordner, args):
    """Gibt Liste von (name, bild) zurück und schreibt ein Kontrollbild."""
    grau = cv2.imread(str(pfad), cv2.IMREAD_GRAYSCALE)
    if grau is None:
        raise ValueError("Bild nicht lesbar")
    winkel = schraeglage(grau)
    if abs(winkel) >= 0.05:
        grau = drehen(grau, winkel, rand=int(np.median(grau)))
    bw = binarisieren(grau)
    hell = hintergrund_ausgleichen(grau)
    stamm = pfad.stem
    teile = []                    # (name, bild in Originalgröße)
    rahmen = []                   # für das Kontrollbild: (x0, y0, x1, y1)

    y_oben, y_unten = textbereich_oben(bw), textbereich_unten(bw)

    if args.modus == "seite":
        teile.append((f"{stamm}_seite", hell[y_oben:y_unten]))
        rahmen.append((0, y_oben, hell.shape[1], y_unten))
    else:
        bander = spaltengrenzen(bw[y_oben:y_unten], args.min_spaltenbreite, args.spalten)
        mitte = bander[len(bander) // 2][2]
        y_start = y_oben + koerper_start(bw[y_oben:y_unten], mitte)
        if y_start > y_oben + 20 and len(mitte) > 2:
            bander = spaltengrenzen(bw[y_start:y_unten], args.min_spaltenbreite, len(mitte) - 1)
        else:
            y_start = y_oben
        bander = [(ya + y_start, yb + y_start, xs) for ya, yb, xs in bander]
        n = len(bander[0][2]) - 1

        if y_start - y_oben > 60:                         # Kopf (seitenbreit)
            teile.append((f"{stamm}_kopf", hell[y_oben:y_start]))
            rahmen.append((0, y_oben, hell.shape[1], y_start))

        mitten = [(ya + yb) / 2 for ya, yb, _ in bander]
        ys = np.arange(y_start, y_unten)
        for i in range(n):
            xl = np.interp(ys, mitten, [xs[i] for _, _, xs in bander]) + args.rand
            xr = np.interp(ys, mitten, [xs[i + 1] for _, _, xs in bander]) - args.rand
            x0, x1 = int(max(0, xl.min())), int(min(hell.shape[1], xr.max()))
            spalten_x = np.arange(x0, x1)
            maske = (spalten_x[None, :] >= xl[:, None]) & (spalten_x[None, :] < xr[:, None])
            spalte = np.where(maske, hell[y_start:y_unten, x0:x1], 255).astype(np.uint8)
            teile.append((f"{stamm}_sp{i + 1}", spalte))
            rahmen.append((x0, y_start, x1, y_unten))
        if n == 1:
            print(f"    ! {stamm}: keine Spalten erkannt – Seite wird in Streifen geteilt")

    if args.fuer == "tesseract":
        # keine Kacheln: Tesseract hat keine Bildgrenzen; jede Spalte ein Bild
        bilder = []
        for name, bild in teile:
            faktor = args.kopf_skalierung if name.endswith("_kopf") else args.skalierung
            bild = linien_entfernen(skalieren(bild, faktor))
            bild[bild > args.weiss_schwelle] = 255     # Papierrauschen -> reines Weiß (kleinere PNGs)
            bilder.append((name, weisser_rand(bild)))
        kontrolle_schreiben(hell, rahmen, ziel, stamm, kontroll_ordner)
        return bilder, winkel, None

    if args.modus == "seite":
        # ganze Seite als EIN Bild: so weit verkleinern, dass beide Grenzen passen
        name, bild = teile[0]
        h, b = bild.shape
        faktor = min(1.0, args.max_kante / max(h, b), math.sqrt(args.max_tokens * 784 / (h * b)))
        while tokens(round(b * faktor), round(h * faktor)) > args.max_tokens:
            faktor *= 0.99
        return [(name, skalieren(bild, faktor))], winkel, kontrolle_schreiben(hell, rahmen, ziel, stamm, kontroll_ordner)

    # skalieren, zu breite Teile (z. B. Kopf) senkrecht teilen, dann kacheln
    kacheln = []
    for name, bild in teile:
        # Kopf (Titel, Inhaltsübersicht) ist groß gedruckt -> stärker verkleinern
        bild = skalieren(bild, args.kopf_skalierung if name.endswith("_kopf") else args.skalierung)
        streifen = [bild]
        if bild.shape[1] > args.max_kante:              # breiter als erlaubt -> Streifen
            k = math.ceil(bild.shape[1] / args.max_kante)
            kanten = np.linspace(0, bild.shape[1], k + 1).astype(int)
            streifen = [bild[:, a:b] for a, b in zip(kanten[:-1], kanten[1:])]
        for s_i, st in enumerate(streifen, start=1):
            for t_i, kachel in enumerate(in_kacheln_teilen(st, args.max_kante, args.max_tokens), start=1):
                zusatz = (f"_s{s_i}" if len(streifen) > 1 else "") + f"_t{t_i}"
                kacheln.append((name + zusatz, kachel))

    kontrolle_schreiben(hell, rahmen, ziel, stamm, kontroll_ordner)
    return kacheln, winkel, None


def kontrolle_schreiben(hell, rahmen, ziel, stamm, kontroll_ordner):
    """Kontrollbild: erkannte Teile (Kopf, Spalten) als Rahmen auf der verkleinerten Seite."""
    kontrolle = cv2.cvtColor(skalieren(hell, 0.2), cv2.COLOR_GRAY2BGR)
    for x0, y0, x1, y1 in rahmen:
        cv2.rectangle(kontrolle, (int(x0 * 0.2), int(y0 * 0.2)), (int(x1 * 0.2), int(y1 * 0.2)),
                      (0, 0, 255), 2)
    cv2.imwrite(str(kontroll_ordner / f"{ziel.name}_{stamm}.jpg"), kontrolle,
                [cv2.IMWRITE_JPEG_QUALITY, 70])


# ---------------------------------------------------------------- Ablauf

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jahr", required=True)
    ap.add_argument("--quelle", default="farbscans_treffer", help="Ordner mit <Jahr>/<Ausgabe>/seite_*.jpg")
    ap.add_argument("--fuer", choices=["tesseract", "vlm"], default="tesseract",
                    help="Aufbereitung für Tesseract (Standard) oder für ein Vision-LLM")
    ap.add_argument("--ziel", help="Ausgabeordner (Standard: farbscans_tesseract bzw. farbscans_vlm)")
    ap.add_argument("--modus", choices=["spalten", "seite"], default="spalten",
                    help="spalten (Standard) oder seite = ganze Seite nur verkleinern (zum Vergleich)")
    ap.add_argument("--skalierung", type=float, default=None,
                    help="Faktor gegenüber dem Scan (Standard: tesseract 1.0, vlm 0.6); "
                         "bei --fuer vlm --modus seite ohne Wirkung")
    ap.add_argument("--kopf-skalierung", type=float, default=None,
                    help="Faktor für den seitenbreiten Kopf (Standard: tesseract 1.0, vlm 0.4)")
    ap.add_argument("--max-kante", type=int, default=2576, help="nur vlm: max. lange Kante je Bild (Sonnet 5.5: 2576)")
    ap.add_argument("--max-tokens", type=int, default=4784, help="nur vlm: max. Bild-Tokens je Bild (Sonnet 5.5: 4784)")
    ap.add_argument("--format", choices=["png", "jpg"], help="Standard: tesseract png (verlustfrei), vlm jpg")
    ap.add_argument("--jpeg-qualitaet", type=int, default=90)
    ap.add_argument("--weiss-schwelle", type=int, default=200,
                    help="nur tesseract: hellere Grauwerte werden reines Weiß (255 = aus)")
    ap.add_argument("--spalten", type=int, help="Spaltenzahl vorgeben statt automatisch")
    ap.add_argument("--min-spaltenbreite", type=int, default=400)
    ap.add_argument("--rand", type=int, default=6, help="Pixel, die an Spaltengrenzen weggeschnitten werden")
    ap.add_argument("--probe", type=int, help="nur die ersten N Seiten bearbeiten (zum Testen)")
    ap.add_argument("--neu", action="store_true", help="schon bearbeitete Seiten erneut bearbeiten")
    args = ap.parse_args()
    tess = args.fuer == "tesseract"
    if args.skalierung is None:
        args.skalierung = 1.0 if tess else 0.6
    if args.kopf_skalierung is None:
        args.kopf_skalierung = 1.0 if tess else 0.4
    args.ziel = args.ziel or ("farbscans_tesseract" if tess else "farbscans_vlm")
    args.format = args.format or ("png" if tess else "jpg")

    quelle = Path(args.quelle) / str(args.jahr)
    seiten = sorted(quelle.glob("*/seite_*.jpg"))
    if args.probe:
        seiten = seiten[:args.probe]
    if not seiten:
        raise SystemExit(f"Keine Seiten unter {quelle}/*/seite_*.jpg gefunden")
    ziel_jahr = Path(args.ziel) / str(args.jahr)
    kontroll_ordner = ziel_jahr / "kontrolle"
    kontroll_ordner.mkdir(parents=True, exist_ok=True)
    print(f"{len(seiten)} Seiten, für {args.fuer}, Modus {args.modus}, Skalierung {args.skalierung}, "
          f"Format {args.format}" + ("" if tess else f", max. {args.max_kante} px / {args.max_tokens} Tokens je Bild"))

    csv_pfad = ziel_jahr / f"bilder_{args.jahr}.csv"
    alte = {}
    if csv_pfad.exists() and not args.neu:
        with open(csv_pfad, encoding="utf-8") as f:
            for z in csv.DictReader(f):
                alte.setdefault(z["seite"], []).append(z)

    zeilen, summe_tokens, summe_bytes = [], 0, 0
    for nr, pfad in enumerate(seiten, start=1):
        ausgabe = pfad.parent.name
        schluessel = f"{ausgabe}/{pfad.name}"
        if schluessel in alte:
            zeilen += alte[schluessel]
            summe_tokens += sum(int(z["tokens"]) for z in alte[schluessel])
            summe_bytes += sum(int(z["bytes"]) for z in alte[schluessel])
            continue
        ordner = ziel_jahr / ausgabe
        ordner.mkdir(parents=True, exist_ok=True)
        try:
            kacheln, winkel, _ = seite_verarbeiten(pfad, ordner, kontroll_ordner, args)
        except Exception as e:                       # eine kaputte Seite stoppt nicht alles
            print(f"  ! {schluessel}: {e}")
            continue
        t_seite = 0
        for name, kachel in kacheln:
            if args.format == "png":
                datei = ordner / f"{name}.png"
                cv2.imwrite(str(datei), kachel, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                datei = ordner / f"{name}.jpg"
                cv2.imwrite(str(datei), kachel, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_qualitaet])
            h, b = kachel.shape
            t = tokens(b, h)
            groesse = datei.stat().st_size
            t_seite += t
            summe_tokens += t
            summe_bytes += groesse
            zeilen.append({"seite": schluessel, "kachel": str(datei), "breite": b, "hoehe": h,
                           "tokens": t, "bytes": groesse})
        print(f"  [{nr}/{len(seiten)}] {schluessel}: {len(kacheln)} Bilder"
              + ("" if tess else f", {t_seite:,} Tokens")
              + (f", Schräglage {winkel:+.1f}°" if abs(winkel) >= 0.05 else ""))

    with open(csv_pfad, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seite", "kachel", "breite", "hoehe", "tokens", "bytes"])
        w.writeheader()
        w.writerows(zeilen)

    n_seiten = len({z["seite"] for z in zeilen})
    print(f"\n{n_seiten} Seiten -> {len(zeilen)} Bilder, {summe_bytes / 1e6:.0f} MB")
    if n_seiten and not tess:
        print(f"Bild-Tokens gesamt: {summe_tokens:,} (≈ {summe_tokens // n_seiten:,} je Seite), "
              f"zzgl. Prompt und Antwort-Tokens")
    print(f"Übersicht: {csv_pfad}\nKontrollbilder: {kontroll_ordner}")


if __name__ == "__main__":
    main()
