import os
import nbformat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

percentage_value = 0.98 # Schwellenwert für die Ähnlichkeit (ändern, falls nötig zwischen 0 und 1. z.B. 0.95 entspricht 95%)
base_path = os.path.dirname(os.path.realpath(__file__)) # prüft alles im Pfad in dem sich das Python script befindet

def read_notebook_code(path):
    # Notebook wird geöffnet und als nbformat geladen
    with open(path, "r", encoding="utf-8") as file:
        nb = nbformat.read(file, as_version=4)
    # NUR aus Codezellen den Inhalt sammeln und die celldata für solution true sein muss. Anschließend zu einem String kombinieren
    code_cells = [cell["source"] for cell in nb.cells if (cell.cell_type == "code" and "nbgrader" in cell.metadata and cell.metadata.nbgrader.solution)]

    return "\n".join(code_cells)

def find_plagiat(base_path):
    # Erstelle eine Liste der Pfade zu allen .ipynb-Dateien im Verzeichnis
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(base_path) for f in filenames if f.endswith(".ipynb")]
    # Anzahl ALLER gefundenen Notebooks zur Überprüfung
    print("Anzahl der .ipynb Dokumente: " + str(len(files)) + "\n ")

    # Iteriere über die Notebooks und liest den Code aus
    documents = [read_notebook_code(file) for file in files]
    # Dokumente in Tfidf Matrix umwandeln mit Hilfe von Tfidf Vectorizer
    tfidf_matrix = TfidfVectorizer().fit_transform(documents)
    # Kosinus Ähnlichkeit zwischen allen Dokumenten
    cosine_sim = cosine_similarity(tfidf_matrix)

    results = {}
    # Jedes Dokument mit allen anderen vergleichen
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            similar = cosine_sim[i, j] # Variable für die Ähnlichkeit
            # Checken ob die Ähnlichkeit über dem Schwellenwert liegt
            if similar > percentage_value:
                # Schlüssel für das result dictionary
                #key = (os.path.relpath(files[i], base_path), os.path.relpath(files[j], base_path))
                key = ('_'.join(os.path.basename(os.path.dirname(os.path.relpath(files[i], base_path))).split('_')[:1]), '_'.join(os.path.basename(os.path.dirname(os.path.relpath(files[j], base_path))).split('_')[:1])) # Entfernt alles nach dem zweiten Unterstrich für die Übersichtlichkeit?
                results[key] = "Potenzielles Plagiat mit Ähnlichkeit: " + str(round(similar * 100, 2)) + "%"
    if results:
        return results
    else:
        return "Keine Plagiate gefunden mit Schwellenwert: " + str(percentage_value * 100) + "%"
    

last_person = None  # Variable für die leere Zeile sobald eine neue Person gelistet wird
plagiat_results = find_plagiat(base_path)

# Unterscheidung ob Dict returned wird oder der String, falls keine Plagiate gefunden worden sind
if isinstance(plagiat_results, str):
    print(plagiat_results)
else:
    for pair, status in plagiat_results.items():
        if pair[0] != last_person:
            print("")
        print(f"{pair[0]} und {pair[1]}: {status}")
        last_person = pair[0]



with open(os.path.join(base_path, "plagiatsliste.txt"), "w") as file:
    if isinstance(plagiat_results, str):
        file.write(plagiat_results)
    else:
        last_person = None
        for pair, status in plagiat_results.items():
            if pair[0] != last_person:
                file.write("\n")
            file.write(f"{pair[0]} und {pair[1]}: {status}\n")
            last_person = pair[0]


import csv

# Verwende den base_path und erstelle eine CSV-Datei darin
with open(os.path.join(base_path, "plagiatsliste.csv"), "w", newline='') as file:
    writer = csv.writer(file, delimiter=';')
    # Schreibe die Kopfzeile der CSV-Datei
    writer.writerow(["Student 1", "Student 2", "Ähnlichkeit"])

    if isinstance(plagiat_results, str):
        # Wenn keine Plagiate gefunden wurden, schreibe eine entsprechende Nachricht in die erste Spalte
        writer.writerow([plagiat_results, "", ""])
    else:
        for pair, status in plagiat_results.items():
            # Extrahiere den Ähnlichkeitswert aus dem Status-String
            similarity = status.split(":")[-1].strip()
            # Schreibe die Daten der beteiligten Studenten und deren Ähnlichkeitswert in die CSV-Datei
            writer.writerow([pair[0], pair[1], similarity])
