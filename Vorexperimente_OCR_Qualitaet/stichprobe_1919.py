#!/usr/bin/env python3
"""
Vorexperiment zur Uebrpruefung der Qualitaet des bestehenden OCR-Digitalisats

Generiert von Opus 5.5 


Stichprobe von Ausgaben des Deutschen Reichsanzeigers (1919) ziehen und
die Tesseract-5-hOCRs der Mikrofilm-Digitalisate (UB Mannheim) dazu laden.
Seiten mit Patent-/Gebrauchsmusterlisten werden automatisch vormarkiert.

Ablauf:
  1. Ausgabenliste 1919 lesen (Nummer + Datum)
  2. Geschichtete Zufallsstichprobe: n Ausgaben pro Monat (fester Seed -> reproduzierbar)
  3. Pro Ausgabe auf der Ausgabenseite Film + erstes Bild ablesen (z. B. 002-8962/0446);
     die Ausgabe reicht bis zum Bild vor dem Start der nächsten Ausgabe
  4. hOCR je Bild laden, Klartext daraus erzeugen, Patentseiten erkennen
  5. manifest.csv: Ausgabe, Datum, Film, Bild, Patent-Score, Treffer, Dateien
  Optional: --bilder lädt zusätzlich das Mikrofilmbild der Patentseiten (JPEG),
            --farbscans die Farbscans der ganzen Ausgabe (URL-Muster ungeprüft).

Benutzung:
  pip install requests
  python stichprobe_1919.py --nur-liste           # nur Stichprobe anzeigen
  python stichprobe_1919.py                       # 12 Ausgaben (1 pro Monat)
  python stichprobe_1919.py --pro-monat 2 --seed 42 --bilder
  python stichprobe_1919.py --ausgaben 100 175    # bestimmte Ausgaben
  python stichprobe_1919.py --ausgaben 100 --nachlauf 30   # 30 Bilder weiter schauen
  python stichprobe_1919.py --film 034-8994 --von 330 --bis 340   # Film-/Bildbereich direkt
  python stichprobe_1919.py --trefferliste 1919_stichprobe_urls.txt  # ganze Patentlisten ab fess-Treffern

Bitte höflich crawlen (Pause zwischen Anfragen).
"""

import argparse
import csv
import random
import re
import time
from collections import defaultdict
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

JAHR = 1919
BASIS = "https://digi.bib.uni-mannheim.de"
LISTE_URL = f"{BASIS}/reichsanzeiger/ausgaben/{JAHR}"
AUSGABE_URL = f"{BASIS}/reichsanzeiger/ausgaben/{JAHR}/5/{{nr}}"
# Tesseract-5-hOCR der Mikrofilme; Ordner je Zeitraum (Filme 001–021 bzw. 022–036)
HOCR_BASIS = f"{BASIS}/reichsanzeiger/ocr/film/tesseract-5.0.0-20211201"
HOCR_ORDNER = [  # (erster Film, letzter Film, Ordner)
    (1, 21, "1919-01-02--1921-12-31---001-021"),
    (22, 36, "1922-01-02--1923-12-31---022-036"),
]


def hocr_url(film, bild, ordner=None):
    """URL des hOCR; Ordner wird aus der Filmnummer abgeleitet (1919–1923),
    für andere Zeiträume mit --hocr-ordner angeben."""
    if not ordner:
        nr = int(film.split("-")[0])
        ordner = next((o for a, b, o in HOCR_ORDNER if a <= nr <= b), None)
        if not ordner:
            raise SystemExit(f"Für Film {film} bitte --hocr-ordner angeben "
                             f"(siehe {HOCR_BASIS}/)")
    return f"{HOCR_BASIS}/{ordner}/{film}/{bild}.hocr"
# Mikrofilmbild über den Bildserver (ohne Größenangabe = volle Auflösung; ungeprüft)
BILD_URL = (f"{BASIS}/reichsanzeiger.fcgi?FIF=%2Freichsanzeiger%2Ffilm%2F{{film}}"
            "%2F{bild}.jp2&CVT=jpeg")
FARBSCAN_URL = (f"{BASIS}/periodika/fileadmin/data/"
                "DeutReunP_856399094_{datum}/max/856399094_{jahr}_{nr}_{seite}.jpg")

HEADERS = {"User-Agent": "BA-Projekt Digital Humanities Uni Leipzig (Stichprobe Reichsanzeiger)"}
PAUSE = 1.0           # Sekunden zwischen Anfragen
MAX_BILDER = 150      # Sicherheitsgrenze pro Ausgabe

# Merkmale einer Patent-/Gebrauchsmusterseite; tolerant gegenüber ſ/s und OCR-Fehlern
PATENT_MUSTER = {
    "Ziffern links": r"Ziffern\s+links",
    "bezeichnen die Klasse": r"bezeichnen\s+die\s+Kla[ſs]{2}e",
    "Gebrauchsmuster": r"Gebrauchsmu[ſs]ter",
    "Patente": r"\bPatent(e|erteilungen|anmeldungen)\b",
    "Patentamt": r"Patentamt",
    "Zentral-Handelsregister-Beilage": r"Zentral\W{0,3}[Hh]andel[sſ]\W{0,2}regi",
}
# Zeilen, mit denen ein Listeneintrag beginnt (tolerant gegenüber OCR-Fehlern):
#   Patent 1923: "18e, 4. G. 59 113. Hannoversche ..."   (Klasse, Gruppe, Aktenzeichen)
#   Patent 1901: "18. Sch. 15 991. Elektrischer ..."
#   Gebrauchsmuster: "7a. 158 080. Vorrichtung ..."
EINTRAG_MUSTER = [
    r"^\W{0,2}\S?\d{0,2}[a-zA-Z]?\s?[,.]\s?\d{1,3}\.\s+[A-ZÄÖÜ][A-Za-z]{0,2}\.?\s?\d{1,3}\s?\d{3}",
    r"^\W{0,2}\S?\d{1,2}[a-z]?\.\s+[A-ZÄÖÜ][a-z]{0,2}\.\s+\d{1,3}\s?\d{3}\.",
    r"^\W{0,2}\S?\d{0,2}[a-zA-Z]?\.\s+\d{3}\s?\d{3}\.\s",
]


class Verbindungsfehler(Exception):
    pass


def neue_session():
    """Session mit automatischen Wiederholungen bei Verbindungsabbrüchen und
    Serverfehlern (Wartezeit wächst: 2, 4, 8, 16, 32 s; Retry-After wird beachtet)."""
    retry = Retry(total=6, connect=6, read=4, status=4, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "HEAD"], respect_retry_after_header=True,
                  raise_on_status=False)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def holen(session, url, **kw):
    try:
        r = session.get(url, headers=HEADERS, timeout=60, **kw)
    except requests.exceptions.RequestException as e:
        raise Verbindungsfehler(f"{url}: {type(e).__name__}: {e}") from None
    time.sleep(PAUSE)
    return r


# ---------------------------------------------------------------- Ausgaben

AUSGABE_LINKS = {}    # nr -> tatsächliche URL der Ausgabenseite (aus der Übersicht)


def ausgaben_lesen(session):
    """Alle verlinkten Ausgaben des Jahres als [(nr, 'YYYYMMDD'), ...].
    Erkennt Links am Linktext "Nr MM-TT" (z. B. "100 05-03"), egal ob die
    URL absolut oder relativ angegeben ist. Gibt [] zurück, wenn nichts gefunden."""
    try:
        r = holen(session, LISTE_URL)
    except Verbindungsfehler as e:
        print(f"! Übersichtsseite nicht erreichbar ({e})")
        return []
    r.encoding = r.encoding or "utf-8"
    html = r.text
    ausgaben = set()
    for href, text in re.findall(r'<a\b[^>]*?href\s*=\s*["\']?([^"\'\s>]+)["\']?[^>]*>(.*?)</a>',
                                 html, re.S | re.I):
        text = unescape(re.sub(r'<[^>]+>', ' ', text))
        m = re.match(r'\s*(\d{1,3})\s+(\d\d)-(\d\d)\s*$', text)
        if not m:
            continue
        nr = int(m.group(1))
        ausgaben.add((nr, f"{JAHR}{m.group(2)}{m.group(3)}"))
        link = urljoin(r.url, unescape(href))
        # nur übernehmen, wenn der Link plausibel auf diese Ausgabe zeigt
        if f"/{JAHR}/" in link and link.rstrip("/").endswith(f"/{nr}"):
            AUSGABE_LINKS[nr] = link
    if not ausgaben:
        debug = Path("debug_uebersicht.html")
        debug.write_text(html, encoding="utf-8")
        print(f"! Übersichtsseite: HTTP {r.status_code}, {len(html)} Zeichen, "
              f"Content-Type {r.headers.get('Content-Type')}")
        print(f"  Keine Ausgaben erkannt. Rohes HTML gespeichert in {debug.resolve()}")
        print("  Anfang:", re.sub(r'\s+', ' ', html[:300]))
    return sorted(ausgaben)


def datum_von_ausgabenseite(session, nr):
    """Fallback: Datum aus der Ausgabenseite lesen ("Show issue 100 (1919-05-03)")."""
    html = holen(session, AUSGABE_URL.format(nr=nr)).text
    m = re.search(rf'\(({JAHR})-(\d\d)-(\d\d)\)', html)
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else None


def stichprobe(ausgaben, pro_monat, seed):
    """Geschichtete Zufallsstichprobe: pro Monat `pro_monat` Ausgaben."""
    rng = random.Random(seed)
    nach_monat = defaultdict(list)
    for nr, datum in ausgaben:
        nach_monat[datum[4:6]].append((nr, datum))
    auswahl = []
    for monat in sorted(nach_monat):
        kandidaten = nach_monat[monat]
        auswahl += rng.sample(kandidaten, min(pro_monat, len(kandidaten)))
    return sorted(auswahl)


def startbild(session, nr):
    """Film und erstes Bild einer Ausgabe von ihrer Ausgabenseite, z. B. ('002-8962', 446)."""
    r = holen(session, AUSGABE_LINKS.get(nr, AUSGABE_URL.format(nr=nr)))
    if r.status_code != 200:
        return None
    html = unescape(r.text).replace("%2F", "/")
    m = re.search(r'/(\d{3}-\d{4})/(\d{4})\.(?:jp2|hocr)', html)
    return (m.group(1), int(m.group(2))) if m else None


def bildbereich(session, nr, alle_nummern):
    """(film, erstes Bild, letztes Bild oder None) einer Ausgabe.
    Ende = Bild vor der nächsten verfügbaren Ausgabe im selben Film."""
    start = startbild(session, nr)
    if not start:
        return None
    film, erstes = start
    for naechste in (n for n in alle_nummern if n > nr):
        folge = startbild(session, naechste)
        if folge:
            if folge[0] == film and folge[1] > erstes:
                return film, erstes, folge[1] - 1
            break
    return film, erstes, None   # Filmende -> laden bis 404


# ---------------------------------------------------------------- hOCR

def hocr_zu_text(hocr):
    """Klartext zeilenweise aus hOCR (Tesseract: ocr_line/ocr_header/... mit ocrx_word)."""
    zeilen = []
    for block in re.split(r"<span[^>]*class=['\"]ocr_(?:line|textfloat|header|caption)['\"]", hocr)[1:]:
        woerter = re.findall(r"class=['\"]ocrx_word['\"][^>]*>(.*?)</span>", block, re.S)
        woerter = [unescape(re.sub(r"<[^>]+>", "", w)).strip() for w in woerter]
        zeile = " ".join(w for w in woerter if w)
        if zeile:
            zeilen.append(zeile)
    return "\n".join(zeilen)


MONATE = ("Januar|Februar|März|Marz|April|Mai|Juni|Juli|August|September|"
          "Oktober|November|Dezember")
KOPF_MUSTER = (r"[BG]?erlin\W{1,3}(Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Sonnabend|Sonntag)\W{0,3}"
               rf"(?:den\s+)?(\d{{1,2}})\W{{0,2}}\s?({MONATE})")


def kopfdatum(text):
    """Datum aus dem Zeitungskopf ("Berlin, Montag, den 5. Mai") – oder ''.
    Nur ein Hinweis: Kopfzeilen sind oft schlecht erkannt."""
    m = re.search(KOPF_MUSTER, text)
    return f"{m.group(1)} {m.group(2)}. {m.group(3)}" if m else ""


def zeichen_pro_zeile(text):
    """Durchschnittliche Zeilenlänge. Normale Spaltenseiten: ca. 20–60 Zeichen.
    Über ~90 hat Tesseract die Spalten zu seitenbreiten Zeilen verschmolzen –
    solche Bilder brauchen eine eigene Spaltentrennung vor dem OCR."""
    zeilen = [z for z in text.split("\n") if z.strip()]
    return round(sum(map(len, zeilen)) / len(zeilen)) if zeilen else 0


SPALTEN_GRENZE = 90

# Beginn des Handelsregisterteils = Ende der Patent-/Gebrauchsmusterlisten
HR_MUSTER = r"\b(?:[Ii]n|Im)\s+(?:unser(?:em|e)?|das|dem)?\W{0,3}(?:[A-Za-zäöü]{0,8}\s)?Handels\W{0,2}regi"
HR_GRENZE = 3


def ist_handelsregister(text):
    return len(re.findall(HR_MUSTER, text)) >= HR_GRENZE


def patent_score(text):
    treffer = [name for name, muster in PATENT_MUSTER.items() if re.search(muster, text)]
    eintraege = sum(1 for zeile in text.split("\n")
                    if any(re.search(m, zeile) for m in EINTRAG_MUSTER))
    # Stichworte zählen stark, viele eintragsähnliche Zeilen zusätzlich
    score = 2 * len(treffer) + min(eintraege // 10, 5)
    return score, treffer, eintraege


# ---------------------------------------------------------------- Download

def bilder_laden(session, film, erstes, ende, ordner, manifest, args, kennung=("", ""),
                 stopp_handelsregister=False):
    """Lädt die hOCRs film/erstes..ende (stoppt bei 404) und bewertet jede Seite.
    Mit stopp_handelsregister endet die Schleife nach dem ersten Bild, auf dem der
    Handelsregisterteil beginnt (dieses Bild wird noch mitgenommen, da die Liste
    auf seiner linken Hälfte enden kann).
    Gibt (letztes geladenes Bild, Zahl vermutlicher Patentseiten) zurück."""
    patentseiten, letztes = 0, None
    for b in range(erstes, ende + 1):
        bild = f"{b:04d}"
        hocr_datei = ordner / f"{film}_{bild}.hocr"
        if hocr_datei.exists():
            hocr = hocr_datei.read_text(encoding="utf-8")
        else:
            r = holen(session, hocr_url(film, bild, args.hocr_ordner))
            if r.status_code == 404:
                break
            r.raise_for_status()
            r.encoding = "utf-8"
            hocr = r.text
            hocr_datei.write_text(hocr, encoding="utf-8")
        letztes = b

        text = hocr_zu_text(hocr)
        hocr_datei.with_suffix(".txt").write_text(text, encoding="utf-8")
        score, treffer, eintraege = patent_score(text)
        ist_patent = score >= 4

        jpg_datei = ""
        if ist_patent:
            patentseiten += 1
            if args.bilder:
                jpg = hocr_datei.with_suffix(".jpg")
                if not jpg.exists():
                    r = holen(session, BILD_URL.format(film=film, bild=bild))
                    if r.ok and r.headers.get("Content-Type", "").startswith("image"):
                        jpg.write_bytes(r.content)
                jpg_datei = str(jpg) if jpg.exists() else "Bild nicht geladen"

        hr = ist_handelsregister(text)
        if args.nachlauf or args.film or args.trefferliste:
            marke = " <- Patentseite?" if ist_patent else ""
            zpz = zeichen_pro_zeile(text)
            if zpz > SPALTEN_GRENZE:
                marke += "  [Spalten verschmolzen]"
            if hr:
                marke += "  [Handelsregister]"
            print(f"    Bild {bild}: Score {score:2d}, {eintraege:3d} Eintragszeilen, Ø{zpz:3d} Z./Zeile, "
                  f"[{kopfdatum(text) or '?'}] {'; '.join(treffer) or '-'}{marke}")
        manifest.writerow([kennung[0], kennung[1], film, bild, b - erstes + 1, score,
                           "; ".join(treffer), eintraege, "ja" if ist_patent else "",
                           str(hocr_datei), jpg_datei])
        if stopp_handelsregister and hr and b > erstes:
            break
    return letztes, patentseiten


def ausgabe_laden(session, nr, datum, bereich, ziel, manifest, args):
    film, erstes, letztes = bereich
    ordner = ziel / f"{datum}_nr{nr:03d}"
    ordner.mkdir(parents=True, exist_ok=True)
    ende = (letztes + args.nachlauf) if letztes else erstes + MAX_BILDER
    bis, patentseiten = bilder_laden(session, film, erstes, ende, ordner, manifest, args,
                                     kennung=(nr, datum))
    print(f"  Nr. {nr} ({datum}): Film {film}, Bilder {erstes}–{bis}, "
          f"{patentseiten} vermutliche Patentseite(n)")
    if args.farbscans:
        farbscans_laden(session, nr, datum, ordner)


def farbscans_laden(session, nr, datum, ordner):
    """Farbscans der ganzen Ausgabe (URL-Muster laut Betreuer, für 1919 ungeprüft)."""
    for fmt in ("{}", "{:02d}", "{:03d}"):
        url = FARBSCAN_URL.format(datum=datum, jahr=JAHR, nr=nr, seite=fmt.format(1))
        if holen(session, url, stream=True).status_code == 200:
            break
    else:
        print(f"    kein Farbscan für Nr. {nr} gefunden")
        return
    for seite in range(1, MAX_BILDER):
        datei = ordner / f"farbscan_{seite:03d}.jpg"
        if datei.exists():
            continue
        r = holen(session, FARBSCAN_URL.format(datum=datum, jahr=JAHR, nr=nr,
                                               seite=fmt.format(seite)))
        if r.status_code == 404:
            print(f"    {seite - 1} Farbscan-Seiten")
            break
        r.raise_for_status()
        datei.write_bytes(r.content)


# ---------------------------------------------------------------- fess-Trefferliste

MONATSNUMMER = {m: i for i, m in enumerate(
    ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
     "September", "Oktober", "November", "Dezember"], start=1)}


def trefferliste_lesen(pfad):
    """Zeilen wie 'April: Mon: https://.../002-8962/0215.hocr' (Monat/Tag optional).
    Gibt [(monat, tag, film, bild), ...] zurück; Zeilen ohne URL werden übersprungen."""
    treffer = []
    for zeile in Path(pfad).read_text(encoding="utf-8").splitlines():
        url = re.search(r"/(\d{3}-\d{4})/(\d{4})\.hocr", zeile)
        if not url:
            continue
        teile = [t.strip() for t in zeile.split(":")]
        monat = teile[0] if teile and teile[0] in MONATSNUMMER else ""
        tag = teile[1] if monat and len(teile) > 1 and re.fullmatch(r"[A-Za-z]{2,3}", teile[1]) else ""
        treffer.append((monat, tag, url.group(1), int(url.group(2))))
    return treffer


def treffer_auswaehlen(treffer, pro_monat, seed):
    """Optional geschichtete Auswahl: pro Monat höchstens `pro_monat` Treffer."""
    if not pro_monat:
        return treffer
    rng = random.Random(seed)
    nach_monat = defaultdict(list)
    for t in treffer:
        nach_monat[t[0]].append(t)
    auswahl = []
    for monat, liste in nach_monat.items():
        auswahl += rng.sample(liste, min(pro_monat, len(liste)))
    return sorted(auswahl, key=lambda t: (MONATSNUMMER.get(t[0], 0), t[2], t[3]))


def trefferliste_laden(session, args):
    treffer = trefferliste_lesen(args.trefferliste)
    leer = sorted({z.split(":")[0].strip() for z in
                   Path(args.trefferliste).read_text(encoding="utf-8").splitlines()
                   if z.strip() and not re.search(r"\.hocr", z)} - {t[0] for t in treffer})
    pro_monat = args.pro_monat if args.pro_monat_gesetzt else 0
    auswahl = treffer_auswaehlen(treffer, pro_monat, args.seed)
    print(f"{len(treffer)} Treffer in {args.trefferliste}, {len(auswahl)} ausgewählt")
    if leer:
        print(f"  Monate ohne Treffer: {', '.join(leer)}")
    for monat, tag, film, bild in auswahl:
        print(f"  {monat or '?':9s} {tag or '?':3s}  Film {film}, ab Bild {bild:04d}")
    if args.nur_liste:
        return

    ziel = Path(args.ziel)
    ziel.mkdir(exist_ok=True)
    with open(ziel / "manifest.csv", "a", newline="", encoding="utf-8") as f:
        manifest = csv.writer(f)
        if f.tell() == 0:
            manifest.writerow(MANIFEST_KOPF)
        for monat, tag, film, bild in auswahl:
            name = f"{MONATSNUMMER.get(monat, 0):02d}_{tag or 'x'}_{film}_{bild:04d}"
            ordner = ziel / "patentlisten" / name
            ordner.mkdir(parents=True, exist_ok=True)
            print(f"\n{monat} {tag} – Film {film}, ab Bild {bild:04d}:")
            try:
                letztes, n = bilder_laden(session, film, bild, bild + args.max_listenbilder - 1,
                                          ordner, manifest, args, kennung=(f"{monat} {tag}", ""),
                                          stopp_handelsregister=True)
                print(f"  -> Bilder {bild:04d}–{letztes:04d}, {n} mit Patent-Merkmalen")
            except Verbindungsfehler as e:
                print(f"  ! abgebrochen nach mehreren Versuchen – {e}")
            f.flush()


# ---------------------------------------------------------------- Hauptprogramm

MANIFEST_KOPF = ["ausgabe", "datum", "film", "bild", "seite_in_ausgabe", "patent_score",
                 "stichworte", "eintragszeilen", "patentseite", "hocr", "bild_datei"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pro-monat", type=int, default=None,
                    help="Ausgaben (bzw. Treffer) pro Monat; Standard 1, bei --trefferliste alle")
    ap.add_argument("--seed", type=int, default=1919)
    ap.add_argument("--ausgaben", type=int, nargs="*", help="bestimmte Ausgabennummern")
    ap.add_argument("--ziel", default="stichprobe_1919")
    ap.add_argument("--nur-liste", action="store_true", help="nur Stichprobe anzeigen")
    ap.add_argument("--bilder", action="store_true", help="Mikrofilmbilder der Patentseiten laden")
    ap.add_argument("--nachlauf", type=int, default=0,
                    help="N Bilder über das berechnete Ausgabenende hinaus laden (Prüfung, wo die Beilagen liegen)")
    ap.add_argument("--trefferliste", help="Textdatei mit fess-Treffer-URLs ('Monat: Tag: URL')")
    ap.add_argument("--max-listenbilder", type=int, default=12,
                    help="höchstens so viele Bilder pro Patentliste laden (Standard 12)")
    ap.add_argument("--film", help="Film direkt laden, z. B. 034-8994 (mit --von/--bis)")
    ap.add_argument("--von", type=int, help="erstes Bild für --film")
    ap.add_argument("--bis", type=int, help="letztes Bild für --film")
    ap.add_argument("--hocr-ordner", help="hOCR-Ordner für Filme außerhalb 1919–1923, "
                    "z. B. 1897-01-02--1903-02-28---001-045")
    ap.add_argument("--farbscans", action="store_true", help="Farbscans der Ausgaben laden")
    args = ap.parse_args()
    args.pro_monat_gesetzt = args.pro_monat is not None
    if args.pro_monat is None:
        args.pro_monat = 1

    session = neue_session()

    if args.trefferliste:
        trefferliste_laden(session, args)
        return

    if args.film:   # Film-/Bildbereich direkt, ohne Ausgabenliste
        if args.von is None:
            raise SystemExit("--film braucht --von (und optional --bis)")
        bis = args.bis if args.bis is not None else args.von
        ordner = Path(args.ziel) / f"film_{args.film}"
        ordner.mkdir(parents=True, exist_ok=True)
        with open(Path(args.ziel) / "manifest.csv", "a", newline="", encoding="utf-8") as f:
            manifest = csv.writer(f)
            if f.tell() == 0:
                manifest.writerow(MANIFEST_KOPF)
            try:
                letztes, n = bilder_laden(session, args.film, args.von, bis, ordner, manifest, args)
                print(f"Film {args.film}, Bilder {args.von}–{letztes}: {n} vermutliche Patentseite(n)")
            except Verbindungsfehler as e:
                print(f"! abgebrochen nach mehreren Versuchen – {e}")
        return

    alle = ausgaben_lesen(session)
    if alle:
        print(f"{len(alle)} verlinkte Ausgaben für {JAHR} gefunden")
        nummern = [nr for nr, _ in alle]
        if args.ausgaben:
            auswahl = [a for a in alle if a[0] in set(args.ausgaben)]
            fehlen = set(args.ausgaben) - {nr for nr, _ in auswahl}
            if fehlen:
                print(f"  ! Nicht in der Übersicht (fehlende Ausgaben?): {sorted(fehlen)}")
        else:
            auswahl = stichprobe(alle, args.pro_monat, args.seed)
    elif args.ausgaben:
        # Fallback ohne Übersichtsseite: Datum direkt von der Ausgabenseite
        print("  -> Fallback: lese die angegebenen Ausgaben einzeln")
        auswahl = []
        for nr in sorted(set(args.ausgaben)):
            try:
                datum = datum_von_ausgabenseite(session, nr)
            except Verbindungsfehler as e:
                raise SystemExit(f"Server nicht erreichbar ({e}). Bitte später erneut versuchen "
                                 "und prüfen, ob die Seite im Browser lädt.")
            if datum:
                auswahl.append((nr, datum))
            else:
                print(f"  ! Nr. {nr}: Ausgabenseite ohne Datum/Bild")
        nummern = sorted(set(range(1, max(args.ausgaben) + 10)))
    else:
        raise SystemExit("Ohne Übersichtsseite keine Zufallsstichprobe möglich – "
                         "bitte debug_uebersicht.html schicken oder --ausgaben angeben.")

    print("Stichprobe:")
    for nr, datum in auswahl:
        print(f"  Nr. {nr:3d}  {datum[6:]}.{datum[4:6]}.{datum[:4]}")
    if args.nur_liste:
        return

    ziel = Path(args.ziel)
    ziel.mkdir(exist_ok=True)
    with open(ziel / "manifest.csv", "a", newline="", encoding="utf-8") as f:
        manifest = csv.writer(f)
        if f.tell() == 0:
            manifest.writerow(MANIFEST_KOPF)
        for nr, datum in auswahl:
            try:
                bereich = bildbereich(session, nr, nummern)
                if not bereich:
                    print(f"  ! Nr. {nr}: Film/Bild nicht gefunden")
                    continue
                ausgabe_laden(session, nr, datum, bereich, ziel, manifest, args)
            except Verbindungsfehler as e:
                print(f"  ! Nr. {nr}: abgebrochen nach mehreren Versuchen – {e}")
                print("    Bereits geladene Dateien bleiben erhalten; einfach später erneut starten.")
            f.flush()


if __name__ == "__main__":
    main()