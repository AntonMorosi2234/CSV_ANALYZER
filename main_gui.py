import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -------- CSV Analysis --------
def analyze_csv(csv_path: Path, head_rows: int = 5):
    """
    Load and analyze a CSV file:
    - Show basic info (rows, columns, dtypes)
    - Show the first few rows
    - Show statistics for numeric columns
    - Show missing values
    Returns:
        df (DataFrame): the loaded data
        summary (str): a text summary of the analysis
    """
    df = pd.read_csv(csv_path)

    summary = []
    summary.append(f"📂 File: {csv_path.name}")
    summary.append(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    summary.append("Columns: " + ", ".join(df.columns))

    summary.append("\n=== COLUMN TYPES ===")
    summary.append(str(df.dtypes))

    summary.append(f"\n=== FIRST {head_rows} ROWS ===")
    summary.append(str(df.head(head_rows)))

    summary.append("\n=== NUMERIC STATISTICS ===")
    summary.append(str(df.describe(include=[np.number]).transpose()))

    summary.append("\n=== MISSING VALUES ===")
    summary.append(str(df.isna().sum()))

    return df, "\n".join(summary)


# -------- GUI Functions --------
def open_file():
    """
    Open a file dialog to select a CSV.
    Run analysis and show results in the text area and plots tab.
    """
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        return
    try:
        df, summary = analyze_csv(Path(file_path))
        # Show summary text
        text_area.delete("1.0", tk.END)
        text_area.insert(tk.END, summary)
        # Show plots
        plot_graphs(df)
        set_status(f"✅ Analysis complete — {Path(file_path).name}")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        set_status("❌ Error during analysis")


def clear_all():
    """
    Clear text area and remove any existing plots.
    """
    text_area.delete("1.0", tk.END)
    for widget in frame_graphs.winfo_children():
        widget.destroy()
    set_status("🧹 Cleared")


def exit_app():
    """
    Close the application.
    """
    root.quit()


def set_status(msg):
    """
    Update the status bar with a message.
    """
    status_var.set(msg)


def plot_graphs(df: pd.DataFrame):
    """
    Create histograms for all numeric columns in the dataset
    and embed them into the 'Graphs' tab.
    """
    # Clear previous plots
    for widget in frame_graphs.winfo_children():
        widget.destroy()

    # Select numeric columns only
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        lbl = tk.Label(frame_graphs, text="⚠️ No numeric columns available for plots")
        lbl.pack()
        return

    # Create one histogram per numeric column
    for col in num_df.columns:
        fig, ax = plt.subplots(figsize=(4, 3))
        num_df[col].plot(
            kind="hist",
            bins=20,
            ax=ax,
            title=f"Histogram — {col}",
            color="#4CAF50"
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        plt.tight_layout()

        # Embed matplotlib figure inside Tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame_graphs)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(side="left", padx=10, pady=10)


# -------- GUI Layout --------
root = tk.Tk()
root.title("📊 CSV Analyzer Advanced")

# Set window size and center it
win_w, win_h = 1200, 800
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
pos_x = int((screen_w / 2) - (win_w / 2))
pos_y = int((screen_h / 2) - (win_h / 2))
root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

# Global font style
root.option_add("*Font", "SegoeUI 10")

# --- Top bar with buttons ---
frame_top = tk.Frame(root, pady=5)
frame_top.pack(fill="x")

btn_open = tk.Button(frame_top, text="📂 Open CSV", command=open_file,
                     width=15, bg="#4CAF50", fg="white")
btn_open.pack(side="left", padx=5)

btn_clear = tk.Button(frame_top, text="🧹 Clear", command=clear_all,
                      width=15, bg="#f0ad4e", fg="white")
btn_clear.pack(side="left", padx=5)

btn_exit = tk.Button(frame_top, text="❌ Exit", command=exit_app,
                     width=15, bg="#d9534f", fg="white")
btn_exit.pack(side="right", padx=5)

# --- Notebook (tab system) ---
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# Tab 1: Summary text
frame_text = tk.Frame(notebook)
notebook.add(frame_text, text="📄 Summary")

text_area = tk.Text(frame_text, wrap=tk.WORD, bg="#fdfdfd")
text_area.pack(fill="both", expand=True)

# Tab 2: Plots
frame_graphs = tk.Frame(notebook, bg="#fafafa")
notebook.add(frame_graphs, text="📊 Plots")

# --- Status bar ---
status_var = tk.StringVar()
status_var.set("Ready ✅")
status_bar = tk.Label(root, textvariable=status_var, bd=1,
                      relief="sunken", anchor="w", bg="#eee")
status_bar.pack(side="bottom", fill="x")

# Run the app
root.mainloop()
