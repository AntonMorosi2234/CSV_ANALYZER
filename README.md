 📊 CSV Analyzer (Tkinter GUI + CLI)

A simple but powerful CSV Analyzer built with Python, Pandas, Matplotlib, and Tkinter.
It allows you to explore datasets quickly with both a command-line interface (CLI) and a desktop graphical interface (GUI).

 Features

* Load and analyze any `.csv` file.
* Show:

  * Number of rows and columns.
  * Column names and data types.
  * First N rows of the dataset.
  * Descriptive statistics for numeric columns.
  * Missing values per column.
  * Correlation matrix (numeric only).
* Save reports automatically in a `reports/` folder:

  * Head of dataset (`head.csv`).
  * Numeric statistics (`describe_numeric.csv`).
  * Missing values (`missing.csv`).
  * Correlation matrix (`correlation.csv`).
  * Histograms for numeric columns as PNG images.
  
  GUI mode (Tkinter):

  * Open CSV file with a file dialog.
  * See summary in a text tab.
  * View histograms directly inside the GUI.
  * Clear and reset analysis with one click.

* CLI mode:

  * Run analysis directly from the terminal with options.

---

 ⚙️ Installation

Clone this repository and install dependencies:

```
git clone https://github.com/yourusername/csv-analyzer.git
cd csv-analyzer
pip install -r requirements.txt
```

Dependencies:

* pandas
* numpy
* matplotlib
* tkinter (usually included with Python)

---

🚀 Usage

Run GUI

```
python main_gui.py
```

* Select a CSV file with the button.
* View results in the GUI.

 Run CLI

```
python main.py --file vendite.csv --report
```

Optional arguments:

* `--file / -f` : Path to CSV file (default: vendite.csv)
* `--sep` : Separator (auto-detected if not given)
* `--encoding` : File encoding (default: utf-8)
* `--head` : Number of rows to preview (default: 5)
* `--report` : Save full report in `reports/`

---

📂 Example Dataset

This repository includes an example file: `vendite.csv` (200 rows).
It is a synthetic sales dataset with the following columns:

* id\_cliente
* nome
* eta
* sesso
* prodotto
* quantita
* prezzo\_unitario
* data\_acquisto

---

 🛠️ Project Structure

```
csv-analyzer/
│
├── main.py          # CLI version
├── main_gui.py      # Tkinter GUI version
├── vendite.csv      # Example dataset (200 rows)
├── reports/         # Generated reports (auto-created)
└── README.md        # Project documentation
```

---

📜 License

This project is licensed under the MIT License.
Feel free to use, modify, and share it.

---

👨‍💻 Author: Anton Morosi
💡 Suggestions and pull requests are welcome!


