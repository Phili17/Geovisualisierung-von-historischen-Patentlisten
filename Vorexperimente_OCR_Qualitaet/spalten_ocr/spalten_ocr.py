#!/usr/bin/env python3
"""
Spaltentrennung + OCR für Seiten des Deutschen Reichsanzeigers (Patentlisten).

Ablauf pro Seitenbild:
  1. Graustufen, adaptive Binarisierung, leichte Schräglage korrigieren
  2. Spaltenzahl und grobe Lage der Lücken aus dem Dichteprofil des Textes
     (Trennlinien vorher entfernt; Spalten etwa gleich breit)
  3. Seite in waagrechte Bänder (~500 px) teilen und die Lücke je Band neu
     suchen -> die Grenzen folgen gebogenen oder schrägen Spalten
  4. Kopf abtrennen: Spaltenteil beginnt nach der letzten Zeile, die eine
     Spaltenlücke kreuzt (Titel, 'Vom Zentral-Handelsregister ... ausgegeben')
  5. Jede Spalte als Maske zwischen stufenlos interpolierten Grenzen ausschneiden
     (keine Nähte), Linienreste entfernen, mit Tesseract erkennen (--psm 4)
  6. Ausgabe: <name>_spalten.png (Kontrollbild mit eingezeichneten Grenzen),
     <name>_spalteN.png (Zuschnitte), <name>_ocr.txt (Text Spalte für Spalte)

Benutzung:
  pip install opencv-python-headless numpy
  # Tesseract 5 + Fraktur-Modell (z. B. frak2021, Fraktur oder deu_latf)
  python spalten_ocr.py seite.jpg --lang frak2021 --ausgabe ergebnis/
  python spalten_ocr.py seite.jpg --nur-spalten      # nur Spalten bestimmen, kein OCR
  python spalten_ocr.py *.jpg --spalten 5            # Spaltenzahl vorgeben

Stellschrauben (siehe --help): --spalten, --min-spaltenbreite, --rand, --psm.
Immer zuerst das Kontrollbild <name>_spalten.png ansehen!
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------- Vorverarbeitung

def binarisieren(grau):
    """Text = 255, Hintergrund = 0. Adaptive Schwelle, damit vergilbte oder
    schattige Bereiche (Bundsteg, Ränder) nicht schwarz werden."""
    grau = cv2.GaussianBlur(grau, (3, 3), 0)
    bw = cv2.adaptiveThreshold(grau, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 15)
    # Einzelpixel-Rauschen entfernen
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def schraeglage(grau, max_grad=2.0, schritt=0.1):
    """Winkel, bei dem das waagrechte Projektionsprofil am 'schärfsten' ist
    (auf einer Otsu-Binarisierung, die dafür stabiler ist als die adaptive)."""
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


def drehen(bild, winkel, rand=255):
    h, w = bild.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), winkel, 1.0)
    return cv2.warpAffine(bild, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderValue=rand if bild.ndim == 2 else (rand,) * 3)


# ---------------------------------------------------------------- Layout

def glaetten(profil, fenster):
    return np.convolve(profil, np.ones(fenster), mode="same")


def waagrechte_linien(bw, min_anteil=0.4):
    """y-Positionen langer waagrechter Linien. Linien dürfen leicht schräg oder
    unterbrochen sein: erkannt werden Stücke ab 4 % Seitenbreite, gezählt wird
    in einem Band von ±6 Pixeln."""
    h, w = bw.shape
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.04), 1))
    linien = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kern)
    zeilen = glaetten(linien.sum(axis=1) / 255, 13)
    ys = np.where(zeilen > w * min_anteil)[0]
    return gruppieren(ys, abstand=20)


def senkrechte_linien(bw, min_anteil=0.25):
    """x-Positionen senkrechter Trennlinien. Linien dürfen leicht gebogen oder
    unterbrochen sein: erkannt werden Stücke ab 2 % Bereichshöhe, gezählt wird
    in einem Band von ±20 Pixeln."""
    h, w = bw.shape
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(h * 0.02))))
    linien = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kern)
    spalten = glaetten(linien.sum(axis=0) / 255, 41)
    xs = np.where(spalten > h * min_anteil)[0]
    return gruppieren(xs, abstand=40)


def gruppieren(werte, abstand=15):
    """Benachbarte Pixelpositionen zu einer Position (Mittelwert) zusammenfassen."""
    gruppen, aktuell = [], []
    for v in werte:
        if aktuell and v - aktuell[-1] > abstand:
            gruppen.append(int(np.mean(aktuell)))
            aktuell = []
        aktuell.append(v)
    if aktuell:
        gruppen.append(int(np.mean(aktuell)))
    return gruppen


def leere_streifen(bw, min_breite=12, schwelle=0.01):
    """Mitten senkrechter Streifen ohne Text (Fallback, wenn keine Trennlinien da sind)."""
    h, w = bw.shape
    profil = bw.sum(axis=0) / 255 / h
    profil = np.convolve(profil, np.ones(9) / 9, mode="same")
    leer = profil < schwelle
    mitten, start = [], None
    for x, l in enumerate(leer):
        if l and start is None:
            start = x
        elif not l and start is not None:
            if x - start >= min_breite:
                mitten.append((start + x) // 2)
            start = None
    return mitten


def kopf_ende(bw, kopf_anteil=0.3):
    """y, ab dem der Spaltenteil beginnt: unter der letzten langen waagrechten Linie
    im oberen Seitenbereich (Kopf, Titel, 'Vom Zentral-Handelsregister ... ausgegeben')."""
    h = bw.shape[0]
    linien = [y for y in waagrechte_linien(bw) if y < h * kopf_anteil]
    return (max(linien) + 10) if linien else 0


def oeffnen_1d(profil, breite):
    """1D-Öffnung (Minimum, dann Maximum): entfernt Spitzen, die schmaler als
    `breite` sind – z. B. dünne, unterbrochene Spaltenlinien in einer Lücke."""
    from numpy.lib.stride_tricks import sliding_window_view
    r = breite // 2
    p = np.pad(profil, r, mode="edge")
    mn = sliding_window_view(p, 2 * r + 1).min(axis=1)
    p = np.pad(mn, r, mode="edge")
    return sliding_window_view(p, 2 * r + 1).max(axis=1)


def textdichte(bw, linienbreite=14):
    """Senkrechtes Dichteprofil des Textes. Lange senkrechte Linien werden
    vorher entfernt, schmale Reste davon per 1D-Öffnung; danach nur leicht
    geglättet, damit schmale Spaltenlücken (~40 px bei 400 dpi) erhalten bleiben."""
    h = bw.shape[0]
    senkrecht = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    text = cv2.subtract(bw, senkrecht)
    profil = text.sum(axis=0) / 255 / max(h, 1)
    profil = oeffnen_1d(profil, linienbreite)
    return glaetten(profil, 5) / 5


def grenzen_fuer_n(profil, links, rechts, n):
    """Bei n gleich breiten Spalten liegt jede Lücke nahe links + i*(Breite/n):
    dort im Fenster ±25 % der Spaltenbreite das Dichteminimum suchen."""
    breite = (rechts - links) / n
    grenzen = [links]
    for i in range(1, n):
        mitte = links + i * breite
        a, b = int(mitte - 0.25 * breite), int(mitte + 0.25 * breite)
        grenzen.append(a + int(np.argmin(profil[a:b])))
    grenzen.append(rechts)
    return grenzen


def baender(hoehe, band_hoehe, bw=None, suchradius=60):
    """Waagrechte Bänder ~band_hoehe hoch. Mit bw werden die Bandkanten in den
    Zeilenzwischenraum gelegt (Zeile mit der wenigsten Tinte im Umkreis), damit
    beim Zusammensetzen keine Textzeile durchschnitten wird."""
    n = max(1, round(hoehe / band_hoehe))
    kanten = np.linspace(0, hoehe, n + 1).astype(int)
    if bw is not None:
        zeilen = glaetten(bw.sum(axis=1) / 255, 5)
        for i in range(1, n):
            a, b = max(0, kanten[i] - suchradius), min(hoehe, kanten[i] + suchradius)
            kanten[i] = a + int(np.argmin(zeilen[a:b]))
    return list(zip(kanten[:-1], kanten[1:]))


def luecke_im_band(profil, x, suchradius):
    a, b = max(0, x - suchradius), min(len(profil), x + suchradius)
    xm = a + int(np.argmin(profil[a:b]))
    return xm, float(profil[xm])


def spaltengrenzen(bw_koerper, min_spaltenbreite, n_vorgabe=None, max_spalten=8,
                   band_hoehe=500):
    """Spaltengrenzen je waagrechtem Band: [(y_a, y_b, [x0, x1, ..., xn]), ...].
    1. grobe Lage der Lücken aus dem Dichteprofil der ganzen Seite (gleich breite Spalten)
    2. je Band die Lücke im Umkreis neu suchen -> folgt gebogenen/schrägen Spalten
    Ohne Vorgabe: größte Spaltenzahl, bei der in mind. 70 % der Bänder jede Lücke
    fast leer ist (Dichte < 10 % des Band-Medians)."""
    h, w = bw_koerper.shape
    profil = textdichte(bw_koerper)
    # Textbereich: breite Bereiche mit Text; schmale Störungen (Scankante,
    # Linien am Rand) werden per 1D-Öffnung über 80 px ignoriert
    breit = oeffnen_1d(profil, 80)
    belegt = np.where(breit > 0.3 * np.median(profil[profil > 0]))[0]
    links, rechts = (int(belegt[0]), int(belegt[-1])) if len(belegt) else (0, w - 1)
    band_profile = [(ya, yb, textdichte(bw_koerper[ya:yb]))
                    for ya, yb in baender(h, band_hoehe, bw_koerper)]

    def bewerten(grob):
        breite = (rechts - links) / (len(grob) - 1)
        radius = int(0.12 * breite)
        ergebnis, gut = [], 0
        for ya, yb, bp in band_profile:
            med = float(np.median(bp[links:rechts])) or 1e-9
            xs, ok = [links], True
            for x in grob[1:-1]:
                xm, d = luecke_im_band(bp, x, radius)
                xs.append(xm)
                ok &= d < 0.10 * med
            xs.append(rechts)
            ergebnis.append((ya, yb, xs))
            gut += ok
        return ergebnis, gut / len(band_profile)

    if n_vorgabe:
        return bewerten(grenzen_fuer_n(profil, links, rechts, n_vorgabe))[0], "Vorgabe"
    beste, quelle = [(0, h, [links, rechts])], "keine Lücken gefunden"
    for n in range(2, max_spalten + 1):
        if (rechts - links) / n < min_spaltenbreite:
            break
        ergebnis, anteil = bewerten(grenzen_fuer_n(profil, links, rechts, n))
        if anteil >= 0.7:
            beste, quelle = ergebnis, f"Dichteprofil, {anteil:.0%} der Bänder sauber"
    return beste, quelle


def koerper_start(bw, xs, max_anteil=0.35, fenster=50, folge=4, spiel=60):
    """Beginn des Spaltenteils: erstes Fenster (im oberen Seitenbereich), ab dem in
    `folge` aufeinanderfolgenden Fenstern zu je `fenster` Zeilen jede Spaltenlücke
    leer ist und in den Spalten Text steht. Die Lücke darf dabei um ±`spiel` px
    von xs abweichen (gebogene Spalten). Überspringt Kopf, Titel und seitenbreite
    Zeilen ('Vom Zentral-Handelsregister ... ausgegeben')."""
    if len(xs) <= 2:
        return 0
    h = bw.shape[0]
    senkrecht = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    n_fenster = int(h * max_anteil) // fenster
    sauber = np.ones(n_fenster, dtype=bool)
    for x in xs[1:-1]:
        streifen = bw[:int(h * max_anteil), max(0, x - spiel):x + spiel]
        # Spaltenlinien (senkrechte Stücke ab 25 px) zählen nicht als Text
        streifen = cv2.subtract(streifen, cv2.morphologyEx(streifen, cv2.MORPH_OPEN, senkrecht))
        for i in range(n_fenster):
            spalten_tinte = streifen[i * fenster:(i + 1) * fenster].sum(axis=0) / 255
            # irgendwo im Spielraum ein 10 px breiter, (fast) leerer Streifen?
            leer = np.convolve(spalten_tinte, np.ones(10), mode="valid").min() <= 3
            sauber[i] &= leer
    # Start = direkt nach dem letzten Fenster, in dem Text eine Lücke kreuzt
    # (seitenbreiter Kopf); danach müssen `folge` saubere Fenster kommen.
    schmutzig = np.where(~sauber)[0]
    if not len(schmutzig):
        return 0
    i = int(schmutzig[-1]) + 1
    return i * fenster if sauber[i:i + folge].all() else 0


# ---------------------------------------------------------------- OCR

def linien_entfernen(grau_ausschnitt):
    """Senkrechte Linienreste (Spaltenlinien am Rand des Zuschnitts) weiß übermalen."""
    bw = binarisieren(grau_ausschnitt)
    linien = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45)))
    linien = cv2.dilate(linien, np.ones((3, 5), np.uint8))
    sauber = grau_ausschnitt.copy()
    sauber[linien > 0] = papierfarbe(grau_ausschnitt)
    return sauber


def papierfarbe(grau):
    """Typischer Grauwert des Papiers (hellere Hälfte der Pixel, Median).
    Wichtig: Flächen nicht reinweiß füllen – sonst setzt Tesseract seine
    Schwelle zwischen Papier und Weiß und hält das ganze Papier für Text."""
    werte = grau[grau < 250]
    if not len(werte):
        return 255
    return int(np.median(werte[werte > np.median(werte)]))


def ocr(bild_pfad, lang, tessdata=None, psm=4):
    cmd = ["tesseract", str(bild_pfad), "stdout", "-l", lang, "--psm", str(psm)]
    if tessdata:
        cmd += ["--tessdata-dir", str(tessdata)]
    erg = subprocess.run(cmd, capture_output=True, text=True)
    if erg.returncode != 0:
        raise SystemExit(f"Tesseract-Fehler: {erg.stderr.strip()}")
    return erg.stdout


def seite_verarbeiten(pfad, args):
    pfad = Path(pfad)
    aus = Path(args.ausgabe)
    aus.mkdir(parents=True, exist_ok=True)
    name = pfad.stem

    farbe = cv2.imread(str(pfad))
    if farbe is None:
        raise SystemExit(f"Bild nicht lesbar: {pfad}")
    grau = cv2.cvtColor(farbe, cv2.COLOR_BGR2GRAY)
    bw = binarisieren(grau)
    winkel = schraeglage(grau)
    if abs(winkel) >= 0.05:
        grau, bw = drehen(grau, winkel), drehen(bw, winkel, rand=0)
        farbe = drehen(farbe, winkel)

    y0 = kopf_ende(bw, args.kopf_anteil)
    koerper = bw[y0:]
    # unteren Rand (leere Zeilen) abschneiden
    zeilen = np.where(koerper.sum(axis=1) / 255 > koerper.shape[1] * 0.01)[0]
    y1 = y0 + (int(zeilen[-1]) + 5 if len(zeilen) else koerper.shape[0])
    # 1. Durchgang: Spalten grob bestimmen, Beginn des Spaltenteils finden
    bander, quelle = spaltengrenzen(bw[y0:y1], args.min_spaltenbreite, args.spalten)
    mitte = bander[len(bander) // 2][2]
    y_start = max(y0, koerper_start(bw, mitte))
    # 2. Durchgang: nur noch auf dem Spaltenteil, mit der gefundenen Spaltenzahl
    if y_start > y0 and len(mitte) > 2:
        bander, quelle = spaltengrenzen(bw[y_start:y1], args.min_spaltenbreite, len(mitte) - 1)
        quelle = "Dichteprofil"
    bander = [(ya + y_start, yb + y_start, xs) for ya, yb, xs in bander]
    n = len(bander[0][2]) - 1
    print(f"{name}: Schräglage {winkel:+.1f}°, Spaltenteil ab y={y_start}, "
          f"{n} Spalten ({quelle}); Grenzen im 1. Band: {bander[0][2]}")

    kontrolle = farbe.copy()
    cv2.line(kontrolle, (0, y_start), (kontrolle.shape[1], y_start), (0, 0, 255), 8)
    for ya, yb, xs in bander:
        for x in xs:
            cv2.line(kontrolle, (x, max(ya, y_start)), (x, yb), (255, 0, 0), 8)
    klein = cv2.resize(kontrolle, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(aus / f"{name}_spalten.png"), klein)

    texte = []
    for i in range(n):
        # Spalte als Maske ausschneiden: linke/rechte Grenze zwischen den
        # Bandmitten linear interpoliert (folgt gebogenen Spalten ohne Nähte)
        mitten = [(ya + yb) / 2 for ya, yb, _ in bander]
        ys = np.arange(y_start, bander[-1][1])
        xl = np.interp(ys, mitten, [xs[i] for _, _, xs in bander]) + args.rand
        xr = np.interp(ys, mitten, [xs[i + 1] for _, _, xs in bander]) - args.rand
        x0, x1 = int(max(0, xl.min())), int(min(grau.shape[1], xr.max()))
        spalten_x = np.arange(x0, x1)
        maske = (spalten_x[None, :] >= xl[:, None]) & (spalten_x[None, :] < xr[:, None])
        ausschnitt = grau[ys[0]:ys[-1] + 1, x0:x1]
        zuschnitt = np.where(maske, ausschnitt, papierfarbe(ausschnitt)).astype(np.uint8)
        zuschnitt = linien_entfernen(zuschnitt)
        datei = aus / f"{name}_spalte{i + 1}.png"
        cv2.imwrite(str(datei), zuschnitt)
        if not args.nur_spalten:
            text = ocr(datei, args.lang, args.tessdata, args.psm)
            texte.append(f"===== Spalte {i + 1} =====\n{text.strip()}\n")
            print(f"  Spalte {i + 1}: {len(text.splitlines())} Zeilen")
    if texte:
        (aus / f"{name}_ocr.txt").write_text("\n".join(texte), encoding="utf-8")
    return bander


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bilder", nargs="+")
    ap.add_argument("--ausgabe", default="spalten_ocr")
    ap.add_argument("--lang", default="Fraktur", help="Tesseract-Modell, z. B. Fraktur, frak2021, deu_latf")
    ap.add_argument("--tessdata", help="Ordner mit .traineddata-Dateien")
    ap.add_argument("--psm", type=int, default=4, help="Tesseract-Seitensegmentierung je Spalte (4 = eine Spalte)")
    ap.add_argument("--spalten", type=int, help="Spaltenzahl vorgeben statt automatisch bestimmen")
    ap.add_argument("--min-spaltenbreite", type=int, default=400,
                    help="Mindestbreite einer Spalte in Pixeln (Standard 400 bei ~400 dpi)")
    ap.add_argument("--kopf-anteil", type=float, default=0.3,
                    help="Kopfbereich wird nur im oberen Anteil der Seite gesucht")
    ap.add_argument("--rand", type=int, default=8, help="Pixel, die an Spaltengrenzen weggeschnitten werden")
    ap.add_argument("--nur-spalten", action="store_true")
    args = ap.parse_args()
    for b in args.bilder:
        seite_verarbeiten(b, args)


if __name__ == "__main__":
    main()
