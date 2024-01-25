Aufgabe
=======

- Teil des Frage-&-Antwort-Moduls:
  - Jeder Frage die zugehörigen (nützlichen) Tags zuordnen
  - ! automatisieren mittels KI-Alg.
- Gegeben: öffentl. verfügbarer Datensatz mit 2 Tabellen:
  - questions (u.a. `Id`, `Title`, `Body`)
  - tags (`Id`, `Tag`)
- Ziel: anhand der Info. in `Title` & `Body` die zugewiesenen Tags vorhersagen

Ansatz 1
========

trainiere einen Transformer (self & cross att.) als einen rekursiven Prädiktor
mit

- input = seq. von `str(Title + "\n" + Body)`
- output = seq. von Tags, Tag für Tag klassifizieren

Modellierung
------------

- Dataflow:
  - transformiere `str(Title + "\n" + Body)` in Seq. von int mittels Tokenizer  
    → einbetten in feature space (token\_dim) durch Embedding-Layer  
      + positional encoding  
    → eingeben in Transformer\_encoder(self att.)  
    ⇒ encoder output: Seq. von (keys & values)
  - generiere LUT für Tag→int (Klassf.-Label < #mögl. tags)
  - berechne Vorhersage für Seq. von Tags rekursiv (d.h., pos. für pos.)
    mittels Transformer\_decoder(self & cross att.) mit Output-Layer:  
    zuletzt generierte Vorhersage  
    → einbetten in token\_dim durch Embedding-Layer  
      + positional encoding  
    → eingeben in Transformer\_decoder  
      dabei: cross att. mit encoder output  
    → tranformieren in Tag-Space durch lin. Output-Layer
- zum Trainieren:
  - cross entropy loss (Tag für Tag)
  - nutze ausgewählte Trainingsdaten (z.B. 5/6 vom Datensatz)

Implementierung des Prototypen (vgl. Programm)
----------------------------------------------

- Datei `pre_processing.py`:
  - generiere pd.df `questions_tags` mittels groupby & join, s.d. `Title` &
    `Body`, `Tag` in einer Tabelle sind
  - generiere pd.series `input_strings` für `str(Title + "\n" + Body)`
  - definiere `tokenizer` für Transf. str→int,
    (hier: Character → int<256, sehr simpel, ! Verbesserung)
  - definiere LUT `tag_to_id` für Transf. Tag→int, und Umkehrung `id_to_tag`
    (hier: #Tags = 16898)
- Datei `NN_train_evaluate.py`:
  - def. NN für Vorhersage, `Transf_Predict_Class`, als Transformer(en+de) mit
    Embbeding & Output layers
  - def. `pos_encode` für positional encoding
  - def. `train_evaluate` für Training mit SGD & Berechnung der Vorhersage für
    Evaluation
- Datei `train_alg.py`:
  - NN initialisieren
  - def. Parameter-Update `optimizer` (hier: ADAM mit lr=0.00001)
  - def. Training-Datensatz  
    (hier: erste 5/6 der gesamten Daten, kann auch zufällig 5/6 wählen)
  - vereinfachte Implementierung mit batch\_size=1 & random sampling:
    - wähle zufällig einen Index für `str(Title + "\n" + Body)` =: `input_str`
      mit zugehörigen Tags =: `input_tags_total`
    - wähle zufällig eine Pos. `pos_predict` in Seq. von Tags zum Vorhersagen
    - nehmne alle Tags vor Pos. `pos_predict` auf =: `input_tags_used` als
      input für decoder von Transformer
    - traininiere NN so, dass nur der Loss bzgl. der letzten Pos. in Output
      berücksichtigt wird (nur Vorhersage für `pos_predict` opt.)
  - Hintergrund für die Vereinfachung mit batch\_size=1 & random sampling:
    - damit Transformer decoder in batch trainiert werden kann, muss eine
      Maskierung von attention weights implementiert werden, damit der
      Prädiktor kausal ist  
      → etwas aufwendiger zu implementieren  
      → Vereinfachung mit batch\_size=1, durch zufälliges Ziehen von
      `pos_predict`
    - Länge von Input für Encoder, `input_str`, var. bei einzelnen Indizes
      (nicht padded wg. Zeit-Beschränkung)  
    (→ ! Verbesserung)
  - Bemerkung: Transformer für NLP anders als für Bildverarbeitung  
    (bei Bild: keine Kausalität, dafür Pos. encoding ggf. aufwendiger)
- Datei `evaluation_routine.py`:
  - def. Test-Datensatz (hier: letze 1/6 der gesamten Daten)
  - für jede Frage & Titel von Testdaten, berechne rekursiv eine Seq. von
    vorhergesagten Tags mit dem trainierten Transformer, wobei Seq.-Länge :=
    max. mögl. #Tags pro Frage (=5)  
    ⇒ #Fehler = symm. diff zw. Label =: `input_tags_total` & Vorhersagen =:
    `predict_class_total`  
      (#(A\B) + #(B\A))  
    ⇒ Fehler-Rate = min{#Fehler / #Tags in label, 1}  
    ⇒ mean error rate über alle Test Daten
  - Tensor `error_rates` für Testdaten gespeichert

Vision zur Weiterentwicklung des Prototypen
-------------------------------------------

- besserer Tokenizer für Preprocessing  
  (z.B. optimierte Kombinationen von Character)
- batch-Training ermöglichen, durch Maskierung für Trainings & Padding von
  input str.
- Weight bei cross entropy := 1 / #(Fragen, die mit Tag-i gelabelt sind)
- Stop-Kriterium für Training durch Überwachung von Validierungsloss
- Hyperparameter tunen, z.B. lr, batch size, NN-Architektur (hier: nur 2
  Trf-Blöcke)
- beim Training Tags zufällig permutieren  
  (nicht sicher, ob die Reihenfolge doch wichtig ist → ausprobieren)

Ansatz 2
========

trainiere einen Transformer mit nur self att. (wie bei GPT) als einen
rekursiven Prädiktor mit

- input = seq. von `str(Title + "\n" + Body)`
- output = seq. von `str(Tag + Tag + ...)`, Token-weise vorhersagen

Modellierung
------------

ähnlich wie bei Ansatz 1, mit nur Encoder-Teil von Transformer &
Klassifizierung für einzelne Tokens statt Tags

Praktische Variante
-------------------

- nutze vortrainiertes Netz z.B. ChatGPT
- entwerfe geeingneten Prompt dafür, z.B.  
  `"please tag the following question:" + "\n" + Title + "\n\n" + Body + "\n" +
  "using one or more of the following tags: python, osx, sql, photoshop, music,
  database, fonts"`  
  (nicht sicher, wie viele Tags man vorschlagen darf)

Vision zur Erweiterung
----------------------

- Optimierung von Prompt anhand der statistischen Auswertung der Resultate  
  (z.B. kleine Variation an Prompt & Fehler-Raten dazu analysieren)
- Vorgang automatisieren mit API (Zugang?)

