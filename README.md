# 🧠 Bayesian Network Builder – Modalità Manuale & Automatica

Questo repository contiene due applicazioni interattive sviluppate in Python per la creazione e l'analisi di **Reti Bayesiane**, realizzate tramite **Jupyter Notebook** e la libreria **pyAgrum**.

Le due modalità disponibili sono:

- **Modalità Manuale**: consente la costruzione personalizzata della rete da parte di un utente esperto.
- **Modalità Automatica**: permette di apprendere la struttura della rete da un dataset utilizzando algoritmi di apprendimento strutturale.

---

## 📁 File contenuti

- `modalità_manuale.py`: interfaccia utente per costruire manualmente una rete bayesiana, definendo variabili, archi, CPT e inferenze.
- `modalità_automatica.py`: applicazione per apprendere automaticamente la struttura della rete da un dataset e applicare inferenza.

---

## 🚀 Funzionalità principali

### 🔧 Modalità Manuale

- Aggiunta di variabili e definizione dei valori.
- Creazione degli archi tra le variabili.
- Inserimento manuale delle CPT (Tabelle di Probabilità Condizionata).
- Visualizzazione del DAG e delle CPT.
- Inferenza esatta a partire da evidenze specifiche.
- Salvataggio e caricamento della rete in formato `.bif`.

### 🤖 Modalità Automatica

- Caricamento di dataset `.xlsx` (es. dataset cardiovascolare).
- Visualizzazione grafica delle variabili.
- Verifica dei dati mancanti.
- Apprendimento della rete tramite Hill Climbing con score AIC, BIC o BDeu.
- Gestione di whitelist e blacklist.
- Calcolo della forza degli archi.
- Visualizzazione delle CPT e inferenza esatta.

---

## 🧰 Librerie necessarie

Assicurarsi di avere installato le seguenti librerie Python:

```bash
pip install pyAgrum ipywidgets matplotlib seaborn pandas
```

---

## 📊 Esempio di dataset

Il file `modalità_automatica.py` utilizza un dataset in formato `.xlsx`, come ad esempio `heart.xlsx`, contenente dati sanitari su pazienti. Assicurarsi che il file sia presente nella stessa directory, modificare la riga 11 dataset = pd.read_excel("heart.xlsx") e sostituire  `heart.xlsx` con il dataset che si vuole usare. Il dataset in formato excel deve essere già discretizzato, attualmente l'applicazione non consente di discretizzare le variabili in automatico.

---

## 📌 Avvio

Avvia **Jupyter Notebook** 

---

## 📚 Riferimenti

Queste applicazioni sono parte di una tesi magistrale dal titolo:

**"Sviluppo di un'applicazione Jupyter per le Reti Bayesiane"**  
Università della Calabria – Corso di Laurea Magistrale in Data Science per le Strategie Aziendali  
A.A. 2024/2025 – Autore: *Eugenio Pascuzzi*

---

## 🪪 Licenza

Questo progetto è distribuito sotto licenza **MIT**.  
Consulta il file [LICENSE](./LICENSE) per i dettagli.
