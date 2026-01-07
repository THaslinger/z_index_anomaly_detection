# Z-Index basierte Anomalieerkennung von CAN-Daten

Dieses Repository enthält eine robuste, statistisch basierte Pipeline zur Anomalieerkennung in CAN-Bus-Daten.
Der implementierte Ansatz ist Teil einer größeren, modularen Anomalieerkennungsarchitektur und fokussiert sich
auf zeitbasierte Abweichungen im Nachrichtenverhalten einzelner CAN-IDs.


---

## Zielstellung

Ziel dieses Projekts ist es, zeitliche Musterabweichungen im Nachrichtenstrom ohne supervision und ohne modellintensive 
Lernverfahren** zu erkennen, indem das normale Sendeverhalten pro CAN-ID statistisch modelliert und gegen dieses 
Referenzverhalten getestet wird.

---

## Grundidee des Ansatzes

- Für jede CAN-ID wird das normale zeitliche Sendeverhalten modelliert
- Abweichungen werden über robuste Z-Scores detektiert
- Die Entscheidung erfolgt Fenster-basiert
- Mehrere heuristische Signale werden aggregiert, um stabile Alarme zu erzeugen

---

## Pipeline-Übersicht (Text-Skizze)

1. **Datenimport**
   - CSV-Import
   - Parsing und Normalisierung der Zeitstempel

2. **Feature Engineering**
   - Berechnung der Inter-Arrival-Times (IAT) pro CAN-ID
   - Windowing auf Basis einer festen Zeitauflösung
   - Aggregation zu Window-Features (Median-IAT, Message-Count)

3. **Robuste Statistik**
   - Iterative Schätzung von Median (`mu`) und MAD-basierter Streuung (`sigma`)
   - Z-Score-Clipping zur Ausreißerunterdrückung
   - Klassifikation von CAN-IDs in „cyclic“ vs. „low_freq“

4. **Anomaliedetektion**
   - Z-Score-basierte Detektion für zyklische IDs
   - Heuristische Regeln für niedrigfrequente IDs

5. **Post-Aggregation**
   - Zeitliche Glättung (konsekutive Anomalien)
   - Mehrheitsentscheidung über mehrere CAN-IDs pro Window

6. **Evaluation**
   - Window-basierte Precision / Recall / F1
   - Vergleich gegen aggregierte Ground Truth

---

## Warum robuste Statistik?

CAN-Daten weisen mehrere Eigenschaften auf, die klassische parametrische Modelle problematisch machen:

- stark schiefe Verteilungen
- sporadische Ausreißer (z. B. Bus-Arbitration, Jitter)
- nichtstationäres Verhalten einzelner IDs

Daher werden:
- Median statt Mittelwert
- MAD statt Standardabweichung
- iteratives Clipping statt einmaliger Schätzung

verwendet.

Das Ergebnis ist ein Modell, das:
- stabil gegenüber Ausreißern ist
- ohne Trainingsphase auskommt
- auch bei kleinen Stichproben sinnvoll funktioniert

---

## Daten

Als Datenquelle dient das Car-Hacking Dataset von Hyun Min Song:  
https://ocslab.hksecurity.net/Datasets/car-hacking-dataset

- Aus den ursprünglich 3,6 Mio. CAN-Nachrichten wurden 10.000 Nachrichten für die Demonstration extrahiert und in public_can2_short.csv bereitgestellt
- Enthält u. a. DoS-, Fuzzy- und Spoofing-Attacken
- Labels werden ausschließlich für die Evaluation genutzt

---

## Limitationen

Dieser Ansatz ist bewusst einfach gehalten und hat bekannte Einschränkungen:

- keine Payload-Analyse (nur Timing)
- feste Window-Größe (kein adaptives Windowing)
- heuristische Schwellenwerte
- keine Online-Adaption der Statistik
- keine Berücksichtigung von CAN-Bus-Topologie oder Signal-Semantik

---

## Motivation für die Code-Auswahl

Dieser Code wurde ausgewählt, da er:
- eine vollständige, in sich geschlossene Pipeline zeigt
- zeigt, dass "einfache" Statistik auch gute Ergebnisse liefern kann und nicht alles ML / DL oder LLM sein muss
- als solide Baseline für weiterführende Arbeiten dient
