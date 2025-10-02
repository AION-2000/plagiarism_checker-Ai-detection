# src/main.py
import tkinter as tk
from plagiarism_checker import PlagiarismCheckerApp

def main():
    root = tk.Tk()
    app = PlagiarismCheckerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()