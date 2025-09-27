# Copyright (C) 2025  Edion Management Systems
import sys
import pandas as pd
import tkinter as tk
from tkinter.font import Font
from tkinter import filedialog, messagebox, Menu, ttk, scrolledtext, simpledialog
from pathlib import Path
import re
from typing import Dict, List, Optional, Union

# TODO:     1. Implement config for regex patterns
#           2. Enhanced TXT parsing with better pattern recognition

SEP = "\n\n\n"
try:
    from utils import SUBJECTS
except ImportError:
    SUBJECTS = ["Computer Science", "Science", "Mathematics", "Social Studies", "History", "Geography",
                "Spanish", "French", "German", "Dutch", "English"]

# Column names for CSV format
CSV_COLUMNS = ["subject", "exercise_type", "grade_level", "difficulty", "modifier", "instruction", "generation"]
STATUS_OPTIONS = ["pending", "pass", "reject"]


class QuestionAnnotator:
    """
    An improved text editor for annotating questions with CSV support and verify mode
    """

    def __init__(self, root: tk.Tk, input_file: str = None, output_file: str = None):
        """Initialize the QuestionAnnotator class.

        Args:
            root: Tkinter root object
            input_file: Path to the input file (CSV or TXT), optional
            output_file: Path to the output file, optional
        """
        self.root = root
        
        # Initialize data variables
        self.filepath_in = input_file
        self.filepath_out = output_file
        self.sessions: List[Dict[str, Union[str, None]]] = []
        self.current_session_idx = 0
        self.verify_mode = False
        
        # UI variables
        self.textbox_font = Font(family="Courier", size=10)
        self.last_replace = None
        
        # Subject regex for TXT parsing
        self.subs = re.compile(r"\((" + "|".join(SUBJECTS) + r")\)")
        
        # Initialize UI
        self.load_window_toolbar()
        self.keybindings()
        
        # Auto-load files if provided
        if self.filepath_in:
            self.load_files(ask_paths=False, warn_user=False)
            
        # Color schemes
        self.color_scheme = "dark_jb"
        self.color_schemes = {
            "default": {
                "bg": "white",
                "fg": "black",
                "textbox_bg": "white",
                "textbox_fg": "black",
                "button_bg": "lightgrey",
                "button_fg": "black",
                "relief": "raised",
                "caret": "black"
            },
            "dark": {
                "bg": "black",
                "fg": "white",
                "textbox_bg": "black",
                "textbox_fg": "white",
                "button_bg": "grey",
                "button_fg": "white",
                "relief": "flat",
                "caret": "white"
            },
            "dark_jb": {
                "bg": "#2B2D30",
                "fg": "#BCBEC4",
                "textbox_bg": "#1E1F22",
                "textbox_fg": "#BCBEC4",
                "button_bg": "#2B2D30",
                "button_fg": "#BCBEC4",
                "relief": "flat",
                "caret": "#CED0D6"
            }
        }
        self.apply_color_scheme()

    def load_window_toolbar(self) -> None:
        self.root.title("Question Annotator - CSV/TXT Editor")
        self.root.geometry("1200x800")

        # Create main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for parameters and controls
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)

        # Right panel for instruction and generation
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=2)

        # Create toolbar menu
        self.create_toolbar_menu()
        
        # Setup left panel
        self.setup_left_panel(left_panel)
        
        # Setup right panel
        self.setup_right_panel(right_panel)
        
        # Status bar
        self.setup_status_bar()

    def create_toolbar_menu(self) -> None:
        """Create the menu bar."""
        self.toolbar = Menu(self.root)
        self.root.config(menu=self.toolbar)

        file_menu = Menu(self.toolbar, tearoff=0)
        file_menu.add_command(label="Load File (F1)", command=self.load_files)
        file_menu.add_command(label="Save Data (F2)", command=self.save_data)
        file_menu.add_separator()
        file_menu.add_command(label="Toggle Verify Mode (F9)", command=self.toggle_verify_mode)
        self.toolbar.add_cascade(label="File", menu=file_menu)

        edit_menu = Menu(self.toolbar, tearoff=0)
        edit_menu.add_command(label="Delete Session (F6)", command=self.delete_session)
        edit_menu.add_command(label="Detect List (F7)", command=self.to_latex)
        edit_menu.add_command(label="Remove parentheses (F8)", command=self.remove_brackets)
        self.toolbar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = Menu(self.toolbar, tearoff=0)
        view_menu.add_command(label="Increase Font Size", command=lambda: self.scale_font_size(1.2))
        view_menu.add_command(label="Decrease Font Size", command=lambda: self.scale_font_size(0.8))
        view_menu.add_command(label="Toggle Color Scheme", command=self.toggle_color_scheme)
        self.toolbar.add_cascade(label="View", menu=view_menu)

        navigate_menu = Menu(self.toolbar, tearoff=0)
        navigate_menu.add_command(label="Previous Session (F3)", command=self.previous_session)
        navigate_menu.add_command(label="Next Session (F4)", command=self.next_session)
        navigate_menu.add_command(label="Jump to Session...", command=self.jump_session)
        self.toolbar.add_cascade(label="Navigate", menu=navigate_menu)

    def setup_left_panel(self, parent: ttk.Frame) -> None:
        """Setup the left panel with parameters and controls."""
        # Parameters frame
        params_frame = ttk.LabelFrame(parent, text="Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Subject
        ttk.Label(params_frame, text="Subject:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.subject_var = tk.StringVar()
        subject_combo = ttk.Combobox(params_frame, textvariable=self.subject_var, values=SUBJECTS, state="readonly")
        subject_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        subject_combo.bind('<<ComboboxSelected>>', self.on_parameter_change)

        # Exercise Type
        ttk.Label(params_frame, text="Exercise Type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.exercise_type_var = tk.StringVar()
        self.exercise_type_entry = ttk.Entry(params_frame, textvariable=self.exercise_type_var)
        self.exercise_type_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        self.exercise_type_entry.bind('<KeyRelease>', self.on_parameter_change)

        # Grade Level
        ttk.Label(params_frame, text="Grade Level:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.grade_level_var = tk.StringVar()
        self.grade_level_entry = ttk.Entry(params_frame, textvariable=self.grade_level_var)
        self.grade_level_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        self.grade_level_entry.bind('<KeyRelease>', self.on_parameter_change)

        # Difficulty
        ttk.Label(params_frame, text="Difficulty:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.difficulty_var = tk.StringVar()
        self.difficulty_entry = ttk.Entry(params_frame, textvariable=self.difficulty_var)
        self.difficulty_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)
        self.difficulty_entry.bind('<KeyRelease>', self.on_parameter_change)

        # Modifier
        ttk.Label(params_frame, text="Modifier:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.modifier_var = tk.StringVar()
        self.modifier_entry = ttk.Entry(params_frame, textvariable=self.modifier_var)
        self.modifier_entry.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=2)
        self.modifier_entry.bind('<KeyRelease>', self.on_parameter_change)

        params_frame.columnconfigure(1, weight=1)

        # Status frame (for verify mode)
        self.status_frame = ttk.LabelFrame(parent, text="Status (Verify Mode)")
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)

        self.status_var = tk.StringVar(value="pending")
        for i, status in enumerate(STATUS_OPTIONS):
            ttk.Radiobutton(self.status_frame, text=status.capitalize(), variable=self.status_var, 
                           value=status, command=self.on_status_change).grid(row=0, column=i, padx=10, pady=5)

        # Verify mode buttons
        buttons_frame = ttk.Frame(self.status_frame)
        buttons_frame.grid(row=1, column=0, columnspan=3, pady=10)

        self.pass_btn = ttk.Button(buttons_frame, text="✓ Pass (Ctrl+P)", command=lambda: self.set_status("pass"))
        self.pass_btn.pack(side=tk.LEFT, padx=5)

        self.reject_btn = ttk.Button(buttons_frame, text="✗ Reject (Ctrl+R)", command=lambda: self.set_status("reject"))
        self.reject_btn.pack(side=tk.LEFT, padx=5)

        self.pending_btn = ttk.Button(buttons_frame, text="? Pending (Ctrl+?)", command=lambda: self.set_status("pending"))
        self.pending_btn.pack(side=tk.LEFT, padx=5)

        # Navigation frame
        nav_frame = ttk.LabelFrame(parent, text="Navigation")
        nav_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(nav_frame, text="Session:").grid(row=0, column=0, padx=5, pady=5)
        self.session_var = tk.StringVar()
        self.session_entry = ttk.Entry(nav_frame, textvariable=self.session_var, width=10)
        self.session_entry.grid(row=0, column=1, padx=5, pady=5)
        self.session_entry.bind("<Return>", lambda _: self.jump_session())

        self.session_label = ttk.Label(nav_frame, text="/ 0")
        self.session_label.grid(row=0, column=2, padx=5, pady=5)

        ttk.Button(nav_frame, text="◀ Previous (F3)", command=self.previous_session).grid(row=1, column=0, padx=2, pady=5)
        ttk.Button(nav_frame, text="▶ Next (F4)", command=self.next_session).grid(row=1, column=1, padx=2, pady=5)

    def setup_right_panel(self, parent: ttk.Frame) -> None:
        """Setup the right panel with instruction and generation text areas."""
        # Instruction frame
        instruction_frame = ttk.LabelFrame(parent, text="Instruction")
        instruction_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.instruction_text = scrolledtext.ScrolledText(instruction_frame, wrap=tk.WORD, font=self.textbox_font, height=10)
        self.instruction_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.instruction_text.bind('<KeyRelease>', self.on_text_change)

        # Generation frame
        generation_frame = ttk.LabelFrame(parent, text="Generation")
        generation_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.generation_text = scrolledtext.ScrolledText(generation_frame, wrap=tk.WORD, font=self.textbox_font, height=10)
        self.generation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.generation_text.bind('<KeyRelease>', self.on_text_change)

    def setup_status_bar(self) -> None:
        """Setup the status bar at the bottom."""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.mode_label = ttk.Label(self.status_bar, text="Mode: Normal")
        self.mode_label.pack(side=tk.LEFT, padx=10)

        self.file_label = ttk.Label(self.status_bar, text="File: None")
        self.file_label.pack(side=tk.LEFT, padx=10)

    def keybindings(self) -> None:
        """Setup keyboard shortcuts."""
        self.root.bind("<F1>", lambda _: self.load_files())
        self.root.bind("<F2>", lambda _: self.save_data())
        self.root.bind("<F3>", lambda _: self.previous_session())
        self.root.bind("<F4>", lambda _: self.next_session())
        self.root.bind("<F6>", lambda _: self.delete_session())
        self.root.bind("<F7>", lambda _: self.to_latex())
        self.root.bind("<F8>", lambda _: self.remove_brackets())
        self.root.bind("<F9>", lambda _: self.toggle_verify_mode())
        
        # Ctrl shortcuts
        self.root.bind("<Control-s>", self.keybinding_event(self.save_data))
        self.root.bind("<Control-o>", self.keybinding_event(self.load_files))
        self.root.bind("<Control-Alt-Up>", self.keybinding_event(self.previous_session))
        self.root.bind("<Control-Alt-Down>", self.keybinding_event(self.next_session))
        
        # Verify mode shortcuts
        self.root.bind("<Control-p>", self.keybinding_event(self.set_status, "pass"))
        self.root.bind("<Control-r>", self.keybinding_event(self.set_status, "reject"))
        self.root.bind("<Control-question>", self.keybinding_event(self.set_status, "pending"))
        
        # Font sizing
        self.root.bind("<Control-plus>", self.keybinding_event(self.scale_font_size, 1.2))
        self.root.bind("<Control-minus>", self.keybinding_event(self.scale_font_size, 0.8))

    @staticmethod
    def keybinding_event(function, *args, **kwargs):
        def dummy_function(event):
            function(*args, **kwargs)
            return "break"
        return dummy_function

    # Data Loading and Saving Methods
    
    def load_files(self, ask_paths: bool = True, warn_user: bool = True) -> None:
        """Load a data file (CSV or TXT format) for annotation.

        Args:
            ask_paths: If True, ask the user for file paths. If False, use class attributes.
            warn_user: If True, show a warning dialog before loading new files.
        """
        if warn_user and self.sessions:
            if not messagebox.askokcancel("Warning", "Loading new files will overwrite any unsaved progress. "
                                                   "Do you want to continue?"):
                return

        if ask_paths:
            self.filepath_in = filedialog.askopenfilename(
                title="Select a data file to read from",
                filetypes=(("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*"))
            )
            if not self.filepath_in:
                return

        try:
            # Determine file type and load accordingly
            file_path = Path(self.filepath_in)
            if file_path.suffix.lower() == '.csv':
                self._load_csv_file()
            elif file_path.suffix.lower() == '.txt':
                self._load_txt_file()
            else:
                # Try to detect format by content
                if self._detect_csv_format():
                    self._load_csv_file()
                else:
                    self._load_txt_file()

            if self.sessions:
                self.current_session_idx = 0
                self._show_session()
                self._update_status_labels()
                self.file_label.config(text=f"File: {Path(self.filepath_in).name}")
                messagebox.showinfo("Success", f"Loaded {len(self.sessions)} sessions from {Path(self.filepath_in).name}")
            else:
                messagebox.showwarning("No Data", "No valid sessions found in the file.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def _detect_csv_format(self) -> bool:
        """Detect if file is in CSV format by examining its content."""
        try:
            with open(self.filepath_in, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                # Check if first line contains expected CSV headers
                expected_headers = set(CSV_COLUMNS)
                line_headers = set([h.strip().lower() for h in first_line.split(',')])
                return len(expected_headers.intersection(line_headers)) >= 4  # At least 4 matching headers
        except Exception:
            return False

    def _load_csv_file(self) -> None:
        """Load data from CSV file using built-in csv module."""
        try:
            data = pd.read_csv(self.filepath_in)
                
                    
            data['status'] = ["pending"] * len(data) if self.verify_mode else ["pass"] * len(data)

            # Replace existing sessions with the CSV rows (each row -> dict)
            self.sessions = data.to_dict(orient='records')
                
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")

    def _load_txt_file(self) -> None:
        """Load and parse data from TXT file format."""
        try:
            with open(self.filepath_in, 'r', encoding='utf-8') as f:
                content = f.read()

            # Regular expression to match headers
            header_pattern = r"\n(?=\(.*?\) \(.*?\)|\(.*?\) T D \d)"

            questions = re.split(header_pattern, content)

            self.sessions = []

            for question in questions:

                header, body = question.strip().split("\n", 1)

                # Create a pattern for splitting
                pattern = r"""
                (\([^)]*\)|S)\s       # Capture group 1: subject
                (\([^)]*\)|T)\s       # Capture group 2: exercise_type  
                (D)\s                 # Capture group 3: "D" (now captured!)
                (\d+|G)\s             # Capture group 4: grade_level
                (\([^)]*\)|M)         # Capture group 5: modifier
                """
                match =re.match(pattern, header, re.VERBOSE)

                params = [
                    p.replace("(", "").replace(")", "") if "(" in p else p for p in match.groups()
                ]

                self.sessions.append(
                    {
                        "subject": params[0], 
                        "exercise_type": params[1], 
                        "difficulty": params[2],  # Now this is "D"
                        "grade_level": params[3], 
                        "modifier": params[4],
                        "generation": body.strip()
                    }
                )
                        
        except Exception as e:
            raise Exception(f"Error loading TXT file: {str(e)}")


    def save_data(self) -> None:
        """Save the current sessions to a CSV file."""
        if not self.sessions:
            messagebox.showwarning("No Data", "No sessions to save.")
            return

        # Update current session before saving
        self._update_current_session()

        # Ask for output file if not provided
        if not self.filepath_out or not messagebox.askyesno("Save Location", 
            f"Save to {self.filepath_out}?\nClick 'No' to choose a different location."):
            
            self.filepath_out = filedialog.asksaveasfilename(
                title="Save annotated data as CSV",
                defaultextension=".csv",
                filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
                initialfile=f"{Path(self.filepath_in).stem}_annotated.csv" if self.filepath_in else "annotated_data.csv"
            )
            if not self.filepath_out:
                return

        try:
            # Save to csv file
            new_data = pd.DataFrame(self.sessions)
            new_data.to_csv(self.filepath_out, index=False)
            
            messagebox.showinfo("Success", f"Saved {len(self.sessions)} sessions to {Path(self.filepath_out).name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")

    # UI Event Handlers
    
    def on_parameter_change(self, event=None) -> None:
        """Handle parameter field changes."""
        if self.sessions and 0 <= self.current_session_idx < len(self.sessions):
            self._update_current_session()

    def on_text_change(self, event=None) -> None:
        """Handle instruction/generation text changes."""
        if self.sessions and 0 <= self.current_session_idx < len(self.sessions):
            self._update_current_session()

    def on_status_change(self, event=None) -> None:
        """Handle status change in verify mode."""
        if self.sessions and 0 <= self.current_session_idx < len(self.sessions):
            self._update_current_session()

    # Session Navigation Methods

    def next_session(self) -> None:
        """Navigate to the next session."""
        if not self.sessions:
            messagebox.showwarning("No Data", "No sessions loaded. Load a file first.")
            return

        if self.current_session_idx < len(self.sessions) - 1:
            self._update_current_session()
            self.current_session_idx += 1
            self._show_session()
        else:
            messagebox.showinfo("End of Data", "This is the last session.")

    def previous_session(self) -> None:
        """Navigate to the previous session."""
        if not self.sessions:
            messagebox.showwarning("No Data", "No sessions loaded. Load a file first.")
            return

        if self.current_session_idx > 0:
            self._update_current_session()
            self.current_session_idx -= 1
            self._show_session()
        else:
            messagebox.showinfo("Beginning of Data", "This is the first session.")

    def jump_session(self) -> None:
        """Jump to a specific session number."""
        if not self.sessions:
            messagebox.showwarning("No Data", "No sessions loaded. Load a file first.")
            return

        try:
            session_num = int(self.session_var.get())
            if 1 <= session_num <= len(self.sessions):
                self._update_current_session()
                self.current_session_idx = session_num - 1
                self._show_session()
            else:
                messagebox.showwarning("Invalid Session", f"Session number must be between 1 and {len(self.sessions)}.")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid session number.")

    def delete_session(self) -> None:
        """Delete the current session."""
        if not self.sessions:
            messagebox.showwarning("No Data", "No sessions loaded. Load a file first.")
            return

        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this session?"):
            self.sessions.pop(self.current_session_idx)
            
            if not self.sessions:
                self._clear_session_display()
                messagebox.showinfo("All Deleted", "All sessions have been deleted.")
            else:
                if self.current_session_idx >= len(self.sessions):
                    self.current_session_idx = len(self.sessions) - 1
                self._show_session()

    # Verify Mode Methods

    def toggle_verify_mode(self) -> None:
        """Toggle between normal and verify mode."""
        self.verify_mode = not self.verify_mode
        mode_text = "Verify" if self.verify_mode else "Normal"
        self.mode_label.config(text=f"Mode: {mode_text}")
        
        # Update status for all sessions if switching to/from verify mode
        for session in self.sessions:
            if not self.verify_mode and session['status'] == 'pending':
                session['status'] = 'pass'
            elif self.verify_mode and session['status'] == 'pass':
                session['status'] = 'pending'
        
        self._show_session()
        
        messagebox.showinfo("Mode Changed", f"Switched to {mode_text} mode")

    def set_status(self, status: str) -> None:
        """Set the status of the current session."""
        if not self.sessions:
            messagebox.showwarning("No Data", "No sessions loaded.")
            return
            
        if status in STATUS_OPTIONS:
            self.status_var.set(status)
            self._update_current_session()
            
            # Auto-advance to next session in verify mode
            if self.verify_mode and self.current_session_idx < len(self.sessions) - 1:
                self.current_session_idx += 1
                self._show_session()

    # Display Methods

    def _show_session(self) -> None:
        """Load the current session data into the UI."""
        if not self.sessions or self.current_session_idx >= len(self.sessions):
            self._clear_session_display()
            return

        session = self.sessions[self.current_session_idx]
        
        # Update parameter fields
        self.subject_var.set(session.get('subject', ''))
        self.exercise_type_var.set(session.get('exercise_type', ''))
        self.grade_level_var.set(session.get('grade_level', ''))
        self.difficulty_var.set(session.get('difficulty', ''))
        self.modifier_var.set(session.get('modifier', ''))
        
        # Update text areas
        self.instruction_text.delete('1.0', tk.END)
        self.instruction_text.insert('1.0', session.get('instruction', ''))
        
        self.generation_text.delete('1.0', tk.END)
        self.generation_text.insert('1.0', session.get('generation', ''))
        
        # Update status
        self.status_var.set(session.get('status', 'pending'))
        
        # Update navigation
        self._update_status_labels()

    def _clear_session_display(self) -> None:
        """Clear all session data from the UI."""
        self.subject_var.set('')
        self.exercise_type_var.set('')
        self.grade_level_var.set('')
        self.difficulty_var.set('')
        self.modifier_var.set('')
        
        self.instruction_text.delete('1.0', tk.END)
        self.generation_text.delete('1.0', tk.END)
        
        self.status_var.set('pending')
        self._update_status_labels()

    def _update_current_session(self) -> None:
        """Update the current session with data from UI fields."""
        if not self.sessions or self.current_session_idx >= len(self.sessions):
            return

        session = self.sessions[self.current_session_idx]
        
        # Update from parameter fields
        session['subject'] = self.subject_var.get().strip()
        session['exercise_type'] = self.exercise_type_var.get().strip()
        session['grade_level'] = self.grade_level_var.get().strip()
        session['difficulty'] = self.difficulty_var.get().strip()
        session['modifier'] = self.modifier_var.get().strip()
        
        # Update from text areas
        session['instruction'] = self.instruction_text.get('1.0', tk.END).strip()
        session['generation'] = self.generation_text.get('1.0', tk.END).strip()
        
        # Update status
        session['status'] = self.status_var.get()

    def _update_status_labels(self) -> None:
        """Update the status labels in the UI."""
        if self.sessions:
            self.session_var.set(str(self.current_session_idx + 1))
            self.session_label.config(text=f"/ {len(self.sessions)}")
        else:
            self.session_var.set("0")
            self.session_label.config(text="/ 0")

    # Utility Methods (keeping some original functionality)

    def apply_color_scheme(self) -> None:
        """Apply the current color scheme to UI elements."""
        scheme = self.color_schemes[self.color_scheme]
        self.root.config(bg=scheme["bg"])
        
        # Apply to text areas
        if hasattr(self, 'instruction_text'):
            self.instruction_text.config(bg=scheme["textbox_bg"], fg=scheme["textbox_fg"],
                                       insertbackground=scheme["caret"])
        if hasattr(self, 'generation_text'):
            self.generation_text.config(bg=scheme["textbox_bg"], fg=scheme["textbox_fg"],
                                      insertbackground=scheme["caret"])

    def toggle_color_scheme(self) -> None:
        """Toggle between color schemes."""
        self.color_scheme = "dark_jb" if self.color_scheme == "default" else "default"
        self.apply_color_scheme()

    def scale_font_size(self, factor: float) -> None:
        """Scale font size by the given factor."""
        self.textbox_font.config(size=int(self.textbox_font["size"] * factor))

    def to_latex(self) -> None:
        """Format selected text or current instruction as LaTeX."""
        # Try to get selected text from instruction area first
        try:
            selected = self.instruction_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            if selected:
                formatted = self._format_text_to_latex(selected)
                self.instruction_text.delete(tk.SEL_FIRST, tk.SEL_LAST)
                self.instruction_text.insert(tk.INSERT, formatted)
                return
        except tk.TclError:
            pass
        
        # If no selection, try generation area
        try:
            selected = self.generation_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            if selected:
                formatted = self._format_text_to_latex(selected)
                self.generation_text.delete(tk.SEL_FIRST, tk.SEL_LAST)
                self.generation_text.insert(tk.INSERT, formatted)
                return
        except tk.TclError:
            pass
        
        messagebox.showwarning("No Selection", "Please select text to format as LaTeX.")

    def _format_text_to_latex(self, text: str) -> str:
        """Format text with LaTeX patterns."""
        # Pattern for numbered lists (e.g., 1. Text 2. Text ...)
        list_pattern = r'(?:\d+\.?\s)([^\d]+)'
        
        # Check if it's a list
        if re.search(r'\d+\.?\s', text):
            items = re.findall(list_pattern, text, re.DOTALL)
            latex_list = "\\begin{enumerate}\n"
            for item in items:
                latex_list += f"\\item {item.strip()}\n"
            latex_list += "\\end{enumerate}"
            return latex_list
        
        # Format other patterns
        # Bold text with double stars (**bold**)
        text = re.sub(r'\*\*(\w+)\*\*', r'\\textbf{\1}', text)
        
        # Italic text with single stars (*italic*)
        text = re.sub(r'\*(\w+)\*', r'\\emph{\1}', text)
        
        # Underlined text with underscores (_underlined_)
        text = re.sub(r'_(\w+)_', r'\\underline{\1}', text)
        
        return text

    def remove_brackets(self) -> None:
        """Remove brackets from selected text or current instruction."""
        # Try instruction area first
        try:
            selected = self.instruction_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            if selected:
                cleaned = re.sub(r'[[\]{}()]', '', selected)
                self.instruction_text.delete(tk.SEL_FIRST, tk.SEL_LAST)
                self.instruction_text.insert(tk.INSERT, cleaned)
                return
        except tk.TclError:
            pass
        
        # Try generation area
        try:
            selected = self.generation_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            if selected:
                cleaned = re.sub(r'[[\]{}()]', '', selected)
                self.generation_text.delete(tk.SEL_FIRST, tk.SEL_LAST)
                self.generation_text.insert(tk.INSERT, cleaned)
                return
        except tk.TclError:
            pass
        
        messagebox.showwarning("No Selection", "Please select text to remove brackets from.")


if __name__ == "__main__":
    root = tk.Tk()
    if len(sys.argv) == 1:
        app = QuestionAnnotator(root)
    elif len(sys.argv) == 3:
        app = QuestionAnnotator(root, sys.argv[1], sys.argv[2])
    else:
        raise ValueError("Invalid number of arguments. Please provide either no arguments or two arguments (filepath_in, filepath_out).")
    root.mainloop()
