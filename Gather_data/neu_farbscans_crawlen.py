#!/usr/bin/env python3
"""
Generiert von Opus 5.5 nach der Aufforderung, ein Python-Skript zu schreiben, das Treffer für
"Ziffern links bezeichnen die Klasse" oder "Schlußziffer hinter dem Komma" oder „gegen unbefugte Benutzung“
in der DDB-Suche finden und nur die Bilder der Treffer crawlen soll.
Erweitert (11.11.2026) um das Crawlen der Folgeseiten, damit die kompletten Patent- und
Gebrauchsmusterlisten geladen werden.



Patentseiten des Deutschen Reichsanzeigers über die DDB-Volltextsuche finden
und die Farbscans der kompletten Patent-/Gebrauchsmusterlisten laden – Jahr für Jahr.

Ablauf je Jahr:
  1. Volltextsuche im Deutschen Zeitungsportal (offizielle DDB-API, Solr-Index
     "newspaper-issues", Seitenebene) nach den Suchphrasen, z. B.
     "Ziffern links bezeichnen die Klasse" – in beiden Schreibweisen (s / ſ)
  2. Folgeseiten prüfen: ab der ersten Trefferseite einer Ausgabe den OCR-Text jeder
     weiteren Seite aus der DDB holen (klein, ohne Bild) und weitergehen, solange er
     nach Patent-/Gebrauchsmusterliste aussieht; Ende, wenn das Handelsregister beginnt
     -> <ziel>/<Jahr>/seiten_<Jahr>.csv (Seite, Rolle, Merkmalszahlen)
  3. Ausgabennummer zum Datum aus der Ausgabenliste der UB Mannheim holen
  4. Farbscans dieser Seiten laden:
       https://digi.bib.uni-mannheim.de/periodika/fileadmin/data/
           DeutReunP_856399094_<JJJJMMTT>/max/856399094_<JJJJ>_<Ausgabe>_<Seite>.jpg
     (Die DDB-Webseite selbst wird nicht abgerufen: sie ist durch eine Bot-Sperre
     geschützt; für Skripte ist die API vorgesehen.)

Benutzung:
  pip install requests beautifulsoup4
  python neu_farbscans_crawlen.py --jahre 1919 --nur-suchen     # Seitenliste erstellen, nichts laden
  python neu_farbscans_crawlen.py --jahre 1919                  # Bilder laden (nutzt die Seitenliste)
  python neu_farbscans_crawlen.py --jahre 1919-1923 --abstand 3 # mehrere Jahre, toleranter suchen

Bitte zuerst mit --nur-suchen prüfen, wie viele Seiten es werden, und langsam laden.
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://api.deutsche-digitale-bibliothek.de/search/index/newspaper-issues/select"
# ZDB-ID "Deutscher Reichsanzeiger und Preußischer Staatsanzeiger" im Zeitungsportal
# (https://www.deutsche-digitale-bibliothek.de/newspaper/2851723-4)
ZDB_ID = "2851723-4"
AUSGABEN_URL = "https://digi.bib.uni-mannheim.de/reichsanzeiger/ausgaben/{jahr}"
BILD_URL = ("https://digi.bib.uni-mannheim.de/periodika/fileadmin/data/"
            "DeutReunP_856399094_{datum}/max/856399094_{jahr}_{nr:03d}_{seite}.jpg")
# Ausgabennummer immer dreistellig (004, 099, 107), Seitenzahl ohne führende Nullen (13)

PHRASEN = [
    "Ziffern links bezeichnen die Klasse",
    "Schlußziffer hinter dem Komma",
    "gegen unbefugte Benutzung",
]
HEADERS = {"User-Agent": "Reichsanzeiger-Forschungsprojekt (Bachelorarbeit Digital Humanities)"}


class Verbindungsfehler(Exception):
    pass


def neue_session():
    retry = Retry(total=6, connect=6, read=4, status=4, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True,
                  raise_on_status=False)
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s


VERSUCHE = 6            # Versuche bei abgebrochener Übertragung
WARTEZEIT = 5           # Sekunden vor dem 2. Versuch, danach jeweils verdoppelt


def holen(session, url, pause, **kw):
    """GET mit Wiederholung, auch wenn die Verbindung mitten in der Antwort
    abbricht (ChunkedEncodingError, Connection reset, Timeout) – das deckt die
    Retry-Einstellung der Session nicht ab. Der Inhalt wird vollständig gelesen,
    bevor die Antwort zurückgegeben wird."""
    warten = WARTEZEIT
    for versuch in range(1, VERSUCHE + 1):
        try:
            r = session.get(url, timeout=(20, 120), **kw)
            _ = r.content            # Antwort komplett empfangen (bricht hier ab, wenn die Leitung reißt)
            time.sleep(pause)
            return r
        except requests.exceptions.RequestException as e:
            fehler = f"{type(e).__name__}: {str(e)[:150]}"
            if versuch == VERSUCHE:
                raise Verbindungsfehler(f"{url}: {fehler}") from None
            print(f"    Verbindung abgebrochen ({type(e).__name__}), Versuch {versuch + 1} "
                  f"von {VERSUCHE} in {warten} s …")
            time.sleep(warten)
            warten *= 2


# ---------------------------------------------------------------- 1. DDB-Suche

def schreibweisen(phrase):
    """Phrase in Antiqua- und Fraktur-Schreibweise (langes ſ), wie sie im OCR vorkommt."""
    lang_s = re.sub(r"s(?=[a-zäöüß])", "ſ", phrase)      # s vor Kleinbuchstaben -> ſ (Klasse -> Klaſſe)
    return sorted({phrase, lang_s})


def suchanfrage(jahr, phrase, abstand, zdb_id):
    teile = [f'plainpagefulltext:"{v}"' + (f"~{abstand}" if abstand else "")
             for v in schreibweisen(phrase)]
    return (f'type:page AND zdb_id:"{zdb_id}" AND '
            f'publication_date:[{jahr}-01-01T00:00:00Z TO {jahr}-12-31T23:59:59Z] AND '
            f'({" OR ".join(teile)})')


def ddb_suchen(session, jahr, phrase, args):
    """Alle Trefferseiten eines Jahres für eine Phrase (Solr-Cursor-Paging).
    Nur die nötigen Felder, ohne Volltext -> kleine Antworten."""
    params = {"q": suchanfrage(jahr, phrase, args.abstand, args.zdb),
              "fl": "id,pagename,publication_date",
              "rows": args.paketgroesse, "sort": "id asc", "cursorMark": "*", "wt": "json"}
    docs = []
    while True:
        r = holen(session, API_URL, args.pause, params=params)
        if r.status_code != 200:
            raise Verbindungsfehler(f"DDB-API: HTTP {r.status_code}: {r.text[:300]}")
        daten = r.json()
        neu = daten["response"]["docs"]
        docs += neu
        if not neu or daten.get("nextCursorMark") in (None, params["cursorMark"]):
            break
        params["cursorMark"] = daten["nextCursorMark"]
    return docs, r.url


def seitenzahl(doc):
    """Seitenzahl einer Trefferseite. Das Feld ist nicht dokumentiert, daher
    mehrere Kandidaten; die DDB-Seiten-ID ist <Ausgaben-ID>-<pagename>."""
    for feld in ("pagenumber", "page_number", "pagenr", "page"):
        wert = doc.get(feld)
        if isinstance(wert, list):
            wert = wert[0] if wert else None
        if wert is not None and str(wert).isdigit():
            return int(wert)
    for wert in (doc.get("pagename"), str(doc.get("id", "")).rsplit("-", 1)[-1]):
        if isinstance(wert, list):
            wert = wert[0] if wert else None
        zahlen = re.findall(r"\d+", str(wert or ""))
        if zahlen:
            return int(zahlen[-1])
    return None


def treffer_aufbereiten(docs_je_phrase):
    """{phrase: [docs]} -> Trefferliste, je Seite einmal, mit allen gefundenen Phrasen."""
    treffer = {}
    for phrase, docs in docs_je_phrase.items():
        for d in docs:
            datum = str(d.get("publication_date", ""))[:10].replace("-", "")
            seite = seitenzahl(d)
            pagename = str(d.get("pagename") or "")
            ausgabe_id = str(d.get("id", ""))
            if pagename and ausgabe_id.endswith("-" + pagename):
                ausgabe_id = ausgabe_id[: -len(pagename) - 1]
            t = treffer.setdefault((datum, seite), {
                "datum": datum, "seite": seite, "ddb_ausgabe": ausgabe_id,
                "ddb_seite": d.get("id"), "phrasen": []})
            if phrase not in t["phrasen"]:
                t["phrasen"].append(phrase)
    for t in treffer.values():
        t["phrasen"] = "; ".join(t["phrasen"])
    return sorted(treffer.values(), key=lambda t: (t["datum"], t["seite"] or 0))


# ---------------------------------------------------------------- 2. Ausgabennummern

def mannheimer_ausgaben(session, jahr, pause):
    """{JJJJMMTT: Ausgabennummer} aus der Ausgabenübersicht der UB Mannheim."""
    r = holen(session, AUSGABEN_URL.format(jahr=jahr), pause)
    soup = BeautifulSoup(r.text, "html.parser")
    tabelle = {}
    for a in soup.find_all("a"):
        m = re.fullmatch(r"\s*(\d{1,3})\s+(\d\d)-(\d\d)\s*", a.get_text(" "))
        if m:
            tabelle.setdefault(f"{jahr}{m.group(2)}{m.group(3)}", int(m.group(1)))
    return tabelle


def ausgabennummer(datum, tabelle):
    """Nummer zum Datum. Fehlt das Datum in der Mannheimer Liste (dort fehlen
    einzelne Ausgaben), wird die Nummer aus den Nachbarn geschätzt, wenn die
    Lücke eindeutig ist (Vorgänger n, Nachfolger n+2 -> n+1)."""
    if datum in tabelle:
        return tabelle[datum], "Liste"
    davor = [d for d in tabelle if d < datum]
    danach = [d for d in tabelle if d > datum]
    if davor and danach:
        n0, n1 = tabelle[max(davor)], tabelle[min(danach)]
        if n1 - n0 == 2:
            return n0 + 1, "geschätzt"
    return None, "unbekannt"


# ---------------------------------------------------------------- 3. Folgeseiten

# Merkmale von Patent-/Gebrauchsmustereinträgen; robust auch bei vermischtem OCR
LISTEN_MERKMALE = re.compile(
    r"Vertr\.|Pat\.?\s?[-⸗]?\s?Anw|Zus\.\s?z\.|G\.\s?m\.\s?b\.\s?H\.|\b\d{2,3}\s?\d{3}\.")
# Beginn des Handelsregisterteils ("In unser Handelsregister ist ... eingetragen")
HR_MERKMALE = re.compile(
    r"\b(?:[Ii]n|Im)\s+(?:unser(?:em|e)?|das|dem)?\W{0,3}(?:[A-Za-zäöü]{0,8}\s)?Handels\W{0,2}regi")


def seitenname(vorlage, n):
    """'FILE_0029_ALTO_DDB_FULLTEXT' + 31 -> 'FILE_0031_ALTO_DDB_FULLTEXT'."""
    m = re.search(r"\d+", vorlage)
    return vorlage[:m.start()] + str(n).zfill(len(m.group())) + vorlage[m.end():]


def seitentext(session, ddb_ausgabe, pagename, args):
    """OCR-Text einer Seite aus der DDB (ca. 30 KB statt 7 MB Bild). None = Seite gibt es nicht."""
    params = {"q": f'id:"{ddb_ausgabe}-{pagename}"', "fl": "id,plainpagefulltext",
              "rows": 1, "wt": "json"}
    r = holen(session, API_URL, args.pause, params=params)
    if r.status_code != 200:
        raise Verbindungsfehler(f"DDB-API: HTTP {r.status_code}")
    docs = r.json()["response"]["docs"]
    if not docs:
        return None
    text = docs[0].get("plainpagefulltext", "")
    return " ".join(text) if isinstance(text, list) else (text or "")


def merkmale(text):
    t = text.replace("ſ", "s")
    return len(LISTEN_MERKMALE.findall(t)), len(HR_MERKMALE.findall(t))


def liste_verfolgen(session, ausgabe, args):
    """Ab der ersten Trefferseite Seite für Seite weitergehen, solange der OCR-Text
    nach Patent-/Gebrauchsmusterliste aussieht. Ende, sobald der Handelsregisterteil
    beginnt oder kaum noch Listenmerkmale da sind (aber nie vor der letzten Trefferseite).
    Rollen: Treffer | Liste | Übergang (Ende, aber mit Listenresten -> wird geladen) | Ende."""
    treffer = sorted(ausgabe["trefferseiten"])
    seiten = []
    for n in range(treffer[0], treffer[0] + args.max_listenseiten):
        text = seitentext(session, ausgabe["ddb_ausgabe"], seitenname(ausgabe["pagename"], n), args)
        if text is None:                       # Ausgabe zu Ende
            break
        liste, hr = merkmale(text)
        if n in treffer:
            rolle = "Treffer"
        elif n < treffer[-1]:
            rolle = "Liste"                    # zwischen zwei Trefferseiten immer mitnehmen
        elif hr >= args.hr_grenze or liste < args.min_merkmale:
            rolle = "Übergang" if liste >= args.min_merkmale // 3 else "Ende"
        else:
            rolle = "Liste"
        seiten.append({"seite": n, "liste": liste, "hr": hr, "rolle": rolle})
        if rolle in ("Übergang", "Ende"):
            break
    return seiten


# ---------------------------------------------------------------- 4. Laden

def bild_laden(session, datum, nr, seite, ordner, args):
    url = BILD_URL.format(datum=datum, jahr=datum[:4], nr=nr, seite=seite)
    ziel = ordner / f"{datum}_nr{nr:03d}" / f"seite_{seite:03d}.jpg"
    if ziel.exists() and ziel.stat().st_size > 0:
        return url, ziel, "schon vorhanden"
    r = holen(session, url, args.pause)
    if r.status_code != 200 or not r.headers.get("Content-Type", "").startswith("image"):
        return url, None, f"HTTP {r.status_code}"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    tmp = ziel.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(ziel)
    return url, ziel, "ok"


# ---------------------------------------------------------------- Ablauf je Jahr

SEITEN_KOPF = ["datum", "wochentag", "ausgabe", "nr_quelle", "seite", "rolle",
               "listenmerkmale", "hr_merkmale", "phrasen", "ddb_ausgabe"]


def jahr_bearbeiten(session, jahr, args):
    ordner = Path(args.ziel) / str(jahr)
    ordner.mkdir(parents=True, exist_ok=True)
    seiten_csv = ordner / f"seiten_{jahr}.csv"
    print(f"\n===== {jahr} =====")

    if seiten_csv.exists() and not args.neu_pruefen:
        # Suche und Seitenprüfung schon erledigt -> direkt laden
        with open(seiten_csv, encoding="utf-8") as f:
            zeilen = list(csv.DictReader(f))
        print(f"Seitenliste vorhanden ({seiten_csv}); mit --neu-pruefen neu erstellen")
    else:
        zeilen = seitenliste_erstellen(session, jahr, ordner, args)
        with open(seiten_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SEITEN_KOPF)
            w.writeheader()
            w.writerows(zeilen)
        print(f"Seitenliste: {seiten_csv}")

    laden = [z for z in zeilen if z["rolle"] in ("Treffer", "Liste", "Übergang")]
    ausgaben = {z["datum"] for z in laden}
    print(f"{len(laden)} Seiten zu laden in {len(ausgaben)} Ausgaben "
          f"(ca. {len(laden) * 7 / 1000:.1f} GB bei ~7 MB/Seite)")
    if args.nur_suchen:
        return

    manifest_pfad = ordner / "manifest.csv"
    kopf = "datum,ausgabe,seite,rolle,url,datei,status"
    if manifest_pfad.exists() and not manifest_pfad.read_text(encoding="utf-8").startswith(kopf):
        manifest_pfad.rename(ordner / "manifest_alte_version.csv")   # Manifest der Vorversion beiseitelegen
    with open(manifest_pfad, "a", newline="", encoding="utf-8") as f:
        manifest = csv.writer(f)
        if f.tell() == 0:
            manifest.writerow(["datum", "ausgabe", "seite", "rolle", "url", "datei", "status"])
        for z in laden:
            if not z["ausgabe"]:
                print(f"  ! {z['datum']} S. {z['seite']}: Ausgabennummer unbekannt – übersprungen")
                manifest.writerow([z["datum"], "", z["seite"], z["rolle"], "", "", "Nummer unbekannt"])
                continue
            datum, nr, seite = z["datum"], int(z["ausgabe"]), int(z["seite"])
            try:
                url, datei, status = bild_laden(session, datum, nr, seite, ordner, args)
            except Verbindungsfehler as e:
                url, datei, status = "", None, "Verbindung abgebrochen"
                print(f"  ! {z['datum']} S. {seite}: {e}")
                print("    (wird beim nächsten Start erneut versucht)")
            manifest.writerow([datum, nr, seite, z["rolle"], url, datei or "", status])
            f.flush()
            hinweis = " (Nummer geschätzt)" if z["nr_quelle"] == "geschätzt" else ""
            print(f"  {datum} Nr. {nr:3d} S. {seite:3d} [{z['rolle']}]: {status}{hinweis}")


def seitenliste_erstellen(session, jahr, ordner, args):
    """Suche + Verfolgung der Folgeseiten -> Zeilen für seiten_<Jahr>.csv."""
    docs_je_phrase, rohdaten = {}, []
    for phrase in args.phrasen:
        docs, anfrage = ddb_suchen(session, jahr, phrase, args)
        docs_je_phrase[phrase] = docs
        rohdaten.append({"phrase": phrase, "anfrage": anfrage, "treffer": docs})
        print(f"  „{phrase}“: {len(docs)} Seiten")
    (ordner / f"ddb_rohdaten_{jahr}.json").write_text(
        json.dumps(rohdaten, ensure_ascii=False, indent=1), encoding="utf-8")
    alle = [d for docs in docs_je_phrase.values() for d in docs]
    if alle and args.felder:
        print("Felder einer Trefferseite:", ", ".join(sorted(alle[0].keys())))

    treffer = [t for t in treffer_aufbereiten(docs_je_phrase) if t["seite"] is not None]
    pagenamen = {d["id"]: d.get("pagename") for d in alle}
    tabelle = mannheimer_ausgaben(session, jahr, args.pause) if treffer else {}

    # nach Ausgaben gruppieren
    ausgaben = {}
    for t in treffer:
        a = ausgaben.setdefault((t["datum"], t["ddb_ausgabe"]), {
            "datum": t["datum"], "ddb_ausgabe": t["ddb_ausgabe"],
            "pagename": pagenamen.get(t["ddb_seite"]) or "FILE_0001_ALTO_DDB_FULLTEXT",
            "trefferseiten": [], "phrasen": {}})
        a["trefferseiten"].append(t["seite"])
        a["phrasen"][t["seite"]] = t["phrasen"]
    print(f"{len(treffer)} Trefferseiten in {len(ausgaben)} Ausgaben – prüfe Folgeseiten …")

    zeilen = []
    for i, a in enumerate(sorted(ausgaben.values(), key=lambda a: a["datum"]), start=1):
        nr, quelle = ausgabennummer(a["datum"], tabelle)
        wt = time.strftime("%a", time.strptime(a["datum"], "%Y%m%d"))
        try:
            seiten = liste_verfolgen(session, a, args)
        except Verbindungsfehler as e:
            print(f"  ! {a['datum']}: {e} – nur Trefferseiten übernommen")
            seiten = [{"seite": n, "liste": "", "hr": "", "rolle": "Treffer"} for n in sorted(a["trefferseiten"])]
        kurz = " ".join(f"{s['seite']}{'' if s['rolle'] in ('Treffer', 'Liste') else '(' + s['rolle'] + ')'}"
                        for s in seiten)
        print(f"  [{i}/{len(ausgaben)}] {a['datum']} {wt} Nr. {nr or '?'}: Seiten {kurz}")
        for s in seiten:
            zeilen.append({"datum": a["datum"], "wochentag": wt, "ausgabe": nr or "", "nr_quelle": quelle,
                           "seite": s["seite"], "rolle": s["rolle"], "listenmerkmale": s["liste"],
                           "hr_merkmale": s["hr"], "phrasen": a["phrasen"].get(s["seite"], ""),
                           "ddb_ausgabe": a["ddb_ausgabe"]})
    return zeilen


def jahre_lesen(angabe):
    jahre = []
    for teil in angabe.split(","):
        a, _, b = teil.strip().partition("-")
        jahre += list(range(int(a), int(b or a) + 1))
    return jahre


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jahre", required=True, help="z. B. 1919  oder  1919-1923  oder  1901,1910")
    ap.add_argument("--phrasen", nargs="+", default=PHRASEN, help="Suchphrasen (Standard: die drei Kopfzeilen)")
    ap.add_argument("--abstand", type=int, default=0,
                    help="erlaubte Wörter zwischen den Phrasenwörtern (Solr-Slop), hilft bei vermischtem OCR")
    ap.add_argument("--zdb", default=ZDB_ID, help=f"ZDB-ID der Zeitung (Standard {ZDB_ID})")
    ap.add_argument("--ziel", default="farbscans_treffer")
    ap.add_argument("--pause", type=float, default=2.0, help="Sekunden zwischen Anfragen")
    ap.add_argument("--paketgroesse", type=int, default=50,
                    help="Treffer pro API-Abfrage (kleiner = robuster bei schlechtem Netz)")
    ap.add_argument("--nur-suchen", action="store_true",
                    help="nur suchen und Folgeseiten prüfen (Seitenliste), keine Bilder laden")
    ap.add_argument("--neu-pruefen", action="store_true",
                    help="Seitenliste neu erstellen, auch wenn seiten_<Jahr>.csv schon existiert")
    ap.add_argument("--min-merkmale", type=int, default=15,
                    help="Mindestzahl an Listenmerkmalen (Vertr., Pat.-Anw., Nummern …) für eine Listenseite")
    ap.add_argument("--hr-grenze", type=int, default=3,
                    help="ab so vielen 'In unser Handelsregister …' gilt die Seite als Handelsregister")
    ap.add_argument("--max-listenseiten", type=int, default=20,
                    help="höchstens so viele Seiten ab der ersten Trefferseite prüfen")
    ap.add_argument("--felder", action="store_true", help="Feldnamen der DDB-Treffer anzeigen")
    args = ap.parse_args()

    session = neue_session()
    for jahr in jahre_lesen(args.jahre):
        try:
            jahr_bearbeiten(session, jahr, args)
        except Verbindungsfehler as e:
            print(f"! {jahr}: {e}")


if __name__ == "__main__":
    main()
