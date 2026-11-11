"""Code generiert mit Gemini 3 und dem Prompt: Schreibe mir ein kurzes Python-Skript, das können soll:
- Volltext (OCR) eines Farbscans (2 Seiten) einlesen (2 Textdateien)
- Eigene Ground Truth dieser Seiten einlesen (zwei vorher händisch transkribierte Textdateien)
- Zeichenfehlerrate (CER) berechnen
"""

import os
from jiwer import cer

def lade_text(dateipfad):
    """Hilfsfunktion zum Einlesen einer Textdatei mit UTF-8-Kodierung."""
    if not os.path.exists(dateipfad):
        raise FileNotFoundError(f"Die Datei '{dateipfad}' wurde nicht gefunden.")
    with open(dateipfad, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # Dateipfade für die 2 Seiten definieren (an deine echten Dateinamen anpassen)
    paare = [
        {"ocr": "Patente_fulltext_1919_101_29.txt", "gt": "gt_seite_1.txt", "name": "Seite 1"},
        {"ocr": "Patente_fulltext_1919_101_30.txt", "gt": "gt_seite_2.txt", "name": "Seite 2"}
    ]
    
    gesamter_gt_text = ""
    gesamter_ocr_text = ""
    
    print("=== CER-Einzelauswertung ===")
    for eintrag in paare:
        try:
            ocr_text = lade_text(eintrag["ocr"])
            gt_text = lade_text(eintrag["gt"])
            
            # CER für diese Seite berechnen
            seiten_cer = cer(gt_text, ocr_text)
            print(f"{eintrag['name']}: {seiten_cer:.4f} ({seiten_cer * 100:.2f}%)")
            
            # Für das Gesamtdokument sammeln
            gesamter_gt_text += gt_text + "\n"
            gesamter_ocr_text += ocr_text + "\n"
            
        except FileNotFoundError as e:
            print(e)

    # Gesamte CER über beide Seiten berechnen
    if gesamter_gt_text and gesamter_ocr_text:
        gesamt_cer = cer(gesamter_gt_text, gesamter_ocr_text)
        print("\n=== Gesamtergebnis ===")
        print(f"Gesamt-CER (über alle Seiten): {gesamt_cer:.4f} ({gesamt_cer * 100:.2f}%)")

if __name__ == "__main__":
    main()