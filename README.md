# Arhitectură Hibridă Deep Learning pentru Prognoza Activelor Financiare

Acest proiect implementează un pipeline avansat de Deep Learning pentru analiza și predicția seriilor de timp financiare masive (>8000 de companii și ETF-uri). Sistemul utilizează o arhitectură neuronală hibridă capabilă să rezolve simultan două sarcini distincte: clasificarea impulsului direcțional (Trend) și regresia variațiilor procentuale pe orizonturi multiple de timp (Preț).

---

## Funcționalități Principale

### Arhitectură Dual-Branch (Multi-Task)

Sistemul procesează datele simultan pe două ramuri specializate:

* **Modul Trend (LSTM):** Evaluează impulsul pieței și prezice direcția următoarei sesiuni (Creștere/Scădere). Optimizat prin `BCEWithLogitsLoss`.
* **Modul Preț (GRU):** Proiectează variația procentuală a prețului (Log Returns) pe orizonturi multiple (ex. 1 zi, 5 zile, 21 zile, 252 zile). Optimizat prin `HuberLoss` pentru a asigura robustețea la volatilitate extremă și anomalii (outliers).

### Optimizare Hardware de Înaltă Performanță

* **TF32 Support:** Suport nativ TensorFloat-32 pentru accelerare pe nucleele NVIDIA Ada Lovelace.
* **Automatic Mixed Precision (AMP):** Utilizare `torch.amp.autocast` și `GradScaler` pentru antrenament mai rapid și o reducere semnificativă a amprentei VRAM.
* **Transfer Asincron:** Optimizarea transferului de memorie CPU-GPU utilizând `pin_memory=True` și `non_blocking=True`.

### Preprocesare Robustă a Datelor

* **Staționaritate:** Transformarea prețurilor absolute în randamente procentuale pentru eliminarea non-staționarității.
* **Standardizare Z-score:** Calculată izolat per activ pe ferestre glisante (Sliding Windows), prevenind strict scurgerile de date (data leakage) din viitor în trecut.

---

## Structura Proiectului

```text
BDT_LLM/
├── dataset/
│   ├── raw_data/          # Folder pentru fișierele .csv brute
│   └── data_loader.py     # Logica de încărcare și preprocesare a datelor
├── models/
│   ├── trend_lstm.py      # Arhitectura modulului de clasificare a trendului
│   ├── prediction_lstm.py # Arhitectura modulului de regresie a prețului
│   ├── checkpoints/       # Salvări intermediare ale ponderilor (la fiecare 5 epoci)
│   └── final_models/      # Modelele complet antrenate exportate aici
├── logs/                  # Fisiere de log pentru monitorizarea antrenamentului
├── global_.py             # Setări și hiperparametri globali
├── main.py                # Punctul de intrare pentru bucla de antrenament
├── predict_symbol.py      # Script interactiv pentru inferență pe un simbol specific
└── .gitignore

```

---

## Cerințe de Sistem & Instalare

**Mediu necesar:**

* **OS:** Linux (Ubuntu) sau WSL2 pe Windows
* **Hardware:** GPU NVIDIA compatibil CUDA (minim 12GB VRAM recomandat pentru Batch Size masiv)
* **Python:** 3.10 sau mai nou

**Instalarea dependențelor:**

```bash
pip install torch pandas numpy scikit-learn tqdm

```

---

## Configurare

Toți hiperparametrii arhitecturii și ai procesului de antrenament sunt extrași și gestionați centralizat în fișierul `global_.py`. Înainte de a rula proiectul, asigurați-vă că ați ajustat următoarele configurații cheie:

* `DATA_DIR`: Calea absolută către folderul principal care conține fișierele `.csv` de tip OHLCV.
* `SEQ_LENGTH`: Fereastra istorică de observare (ex. `60` pentru a observa un trimestru de date de tranzacționare).
* `HORIZONS`: Lista cu orizonturile de predicție în zile (ex. `[1, 5, 21, 252]`).
* `BATCH_SIZE`: Volumul de date procesat simultan (ex. `2048`). Reglați în funcție de VRAM-ul disponibil.
* `HIDDEN_SIZE` & `NUM_LAYERS`: Controlează complexitatea structurală a rețelelor recurente (LSTM/GRU).

---

## Utilizare

### 1. Pregătirea Datelor

Plasați fișierele `.csv` ce conțin datele financiare (format OHLCV) în calea specificată de variabila `DATA_DIR` din `global_.py`. Scriptul va scana automat și recursiv prin toate subfolderele (ex. `/stocks/`, `/etfs/`).

### 2. Declanșarea Antrenamentului

Lansați procesul principal de antrenament:

```bash
python3 main.py

```

*Note: Scriptul va afișa un raport detaliat privind rata de procesare. Ponderile intermediare vor fi salvate automat la fiecare 5 epoci în `models/checkpoints/`. La finalizarea antrenamentului, modelele finale vor fi exportate în `models/final_models/`.*

### 3. Rularea Inferenței (Predicție)

Pentru a evalua un simbol specific folosind modelele antrenate, rulați:

```bash
python3 field_testing.py

```

Acesta este un script interactiv. Veți fi solicitat să introduceți un ticker valid (ex. `AAPL`, `TSLA`). Scriptul va:

1. Localiza istoricul relevant.
2. Scala intrările utilizând doar date istorice contextuale.
3. Returna predicția asamblată: **Direcția estimată** (Trend) și **Nivelurile de Preț** pe orizonturile configurate.
