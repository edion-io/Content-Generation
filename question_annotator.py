# Copyright (C) 2024  Edion Management Systems
import sys
import tkinter as tk
from tkinter.font import Font
from tkinter import filedialog, messagebox, Menu
import re

# TODO:     1. Implement config for chunk size and regex
#           2. Automated header (+ footer?) annotation

SEP = "\n\n\n"

from utils import SUBJECTS


class TextEditor:
    """
    A simple text editor for annotating questions in the specified format
    """

    def __init__(self, root: tk.Tk, input_file: str = None, output_file: str = None):
        """Initialize the TextEditor class.

        Args:
            root: Tkinter root object
            input_file: Path to the input file, optional
            output_file: Path to the output file, optional
        """
        self.textbox = None
        self.root = root
        # Initialize variables
        self.filepath_in = input_file
        self.file_in = None
        self.filepath_out = output_file
        self.file_out = None
        self.textbox_font = Font(family="Courier", size=10)
        self.chunk_size = 100  # lines per chunk
        self.start_chunk = 0
        self.current_chunk = 0
        self.chunks = []
        self.current_section = 0
        self.last_viewed_section = 0
        self.sections = []
        self.current_subject = ""
        self.subs = re.compile(r"\(?(" + "|".join(SUBJECTS) + ")")

        self.load_window_toolbar()
        self.keybindings()
        if self.filepath_in and self.filepath_out:
            self.load_files(ask_paths=False, warn_user=False)

    def load_window(self) -> None:
        self.root.title("Text Section Editor")

        # Create widgets
        self.textbox = tk.Text(self.root, wrap='word')
        self.textbox.pack(expand=1, fill='both')

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(fill='x')

        self.load_button = tk.Button(self.button_frame, text="Load File (F1)", command=self.load_files)
        self.load_button.pack(side='left')

        self.save_button = tk.Button(self.button_frame, text="Save Sections (F2)", command=self.save_sections)
        self.save_button.pack(side='left')

        self.prev_button = tk.Button(self.button_frame, text="Previous Section (F3)", command=self.previous_section)
        self.prev_button.pack(side='left')

        self.next_button = tk.Button(self.button_frame, text="Next Section (F4)", command=self.next_section)
        self.next_button.pack(side='left')

        self.type_button = tk.Button(self.button_frame, text="Detect Type (F5)", command=self.detect_type)
        self.type_button.pack(side='left')

        self.type_button = tk.Button(self.button_frame, text="Delete Section (F6)", command=self.delete_section)
        self.type_button.pack(side='left')

        self.type_button = tk.Button(self.button_frame, text="Next Subject", command=self.next_subject)
        self.type_button.pack(side='left')

        self.jump_chunk_button = tk.Button(self.button_frame, text="Jump To Chunk", command=self.jump_chunk)
        self.jump_chunk_button.pack(side='right')

        self.chunk_number_label = tk.Label(self.button_frame, text="/ -")
        self.chunk_number_label.pack(side='right')

        self.chunk_entry = tk.Entry(self.button_frame, width=8)
        self.chunk_entry.pack(side='right')

        self.chunk_label = tk.Label(self.button_frame, text="Chunk:")
        self.chunk_label.pack(side='right')

    def load_window_toolbar(self):
        self.root.title("Text Section Editor")

        # Create the main textbox
        self.textbox = tk.Text(self.root, wrap='word', font=self.textbox_font)
        self.textbox.pack(expand=1, fill='both')

        # Create a toolbar at the top
        self.toolbar = Menu(self.root)
        self.root.config(menu=self.toolbar)

        file_menu = Menu(self.toolbar, tearoff=0)
        file_menu.add_command(label="Load File (F1)", command=self.load_files)
        file_menu.add_command(label="Save Sections (F2)", command=self.save_sections)
        self.toolbar.add_cascade(label="File", menu=file_menu)

        edit_menu = Menu(self.toolbar, tearoff=0)
        edit_menu.add_command(label="Detect Type (F5)", command=self.detect_type)
        edit_menu.add_command(label="Delete Section (F6)", command=self.delete_section)
        edit_menu.add_command(label="Detect List (F7)", command=self.to_latex)
        edit_menu.add_command(label="Remove parentheses (F8)" , command=self.remove_brackets)
        self.toolbar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = Menu(self.toolbar, tearoff=0)
        view_menu.add_command(label="Increase Font Size", command=lambda:self.scale_font_size(1.2))
        view_menu.add_command(label="Decrease Font Size", command=lambda:self.scale_font_size(0.8))
        self.toolbar.add_cascade(label="View", menu=view_menu)

        navigate_menu = Menu(self.toolbar, tearoff=0)
        navigate_menu.add_command(label="Previous Section (F3)", command=self.previous_section)
        navigate_menu.add_command(label="Next Section (F4)", command=self.next_section)
        navigate_menu.add_command(label="Next Subject", command=self.previous_section)
        navigate_menu.add_command(label="Jump to Chunk ...", command=self.jump_chunk)
        self.toolbar.add_cascade(label="Navigate", menu=navigate_menu)

        # create footer
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(fill='x')

        self.chunk_number_label = tk.Label(self.button_frame, text="/ -")
        self.chunk_number_label.pack(side='right')

        self.chunk_entry = tk.Entry(self.button_frame, width=8)
        self.chunk_entry.bind("<Return>", lambda _: self.jump_chunk())
        self.chunk_entry.pack(side='right')

        # Add chunk and section selectors
        self.chunk_label = tk.Label(self.button_frame, text="Chunk:")
        self.chunk_label.pack(side='right')

    def keybindings(self) -> None:
        self.root.bind("<F1>", lambda _: self.load_files())
        self.root.bind("<F2>", lambda _: self.save_sections())
        self.root.bind("<F3>", lambda _: self.previous_section())
        self.root.bind("<F4>", lambda _: self.next_section())
        self.root.bind("<F5>", lambda _: self.detect_type())
        self.root.bind("<F6>", lambda _: self.delete_section())
        self.root.bind("<F7>", lambda _: self.to_latex())
        self.root.bind("<F8>", lambda _: self.remove_brackets())
        self.root.bind("<Control-s>", self.keybinding_event(self.save_sections))
        self.root.bind("<Control-o>", self.keybinding_event(self.load_files))
        self.root.bind("<Control-Alt-Left>", self.keybinding_event(self.previous_section))
        self.root.bind("<Control-Alt-Right>", self.keybinding_event(self.next_section))
        self.textbox.bind("<Control-l>", self.keybinding_event(self.to_latex))
        self.textbox.bind("<Control-b>", self.keybinding_event(self.apply_bold))
        self.textbox.bind("<Control-i>", self.keybinding_event(self.apply_italic))
        self.textbox.bind("<Control-u>", self.keybinding_event(self.apply_underline))
        self.textbox.bind("<Control-t>", self.keybinding_event(self.detect_type))
        self.textbox.bind("<Control-plus>", self.keybinding_event(lambda:self.scale_font_size(1.2)))
        self.textbox.bind("<Control-minus>", self.keybinding_event(lambda:self.scale_font_size(0.8)))


    @staticmethod
    def keybinding_event(function):
        def dummy_function(event):
            function()
            return "break"
        return dummy_function


    def load_files(self, ask_paths=True, warn_user=True) -> None:
        """Load a text file to read from and a text file to write to.

        Args:
            ask_paths: If True, ask the user for the paths. If False, use class attributes.
            warning: If True, show a warning dialog before loading a new file.

        Returns:

        """
        if warn_user:
            if self.chunks and not messagebox.askokcancel("Warning", "Loading new files will overwrite any unsaved progress. "
                                                                     "Do you want to continue?"):
                return

        if ask_paths:
            self.filepath_in = filedialog.askopenfilename(title="Select a Text File to read from",
                                                          filetypes=(("Text Files", "*.txt"),))
            if not self.filepath_in:
                return

            self.filepath_out = filedialog.askopenfilename(title="Select a Text File to write to",
                                                           filetypes=(("Text Files", "*.txt"),))
            if not self.filepath_out:
                return

        with open(self.filepath_in, 'r', encoding="utf8") as file:
            self.chunks = []
            text_chunk = ""
            count = 0
            for line in file:
                text_chunk += line
                count += 1
                if count == self.chunk_size:
                    self.chunks.append(text_chunk)
                    text_chunk = ""
                    count = 0

        self.current_chunk = 0
        self.current_section = 0
        self._load_sections()
        self._show_section()
        self.chunk_number_label.config(text=f"/ {len(self.chunks)}")

    def save_sections(self) -> None:
        """
        Saves loaded sections to file

        Returns: None
        """

        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to save. Load a file first.")
            return

        self._update_section()

        with open(self.filepath_out, 'w', encoding="utf8") as file:
            for section in self.sections[:self.last_viewed_section+1]:
                while section and section[-1] == "\n":
                    section = section[:-1]
                file.write(section + f"\n\n\n")

        messagebox.showinfo("Saved Sections", f"Saved {self.last_viewed_section + 1} sections in chunks "
                                              f"{self.start_chunk} to {self.current_chunk}")

    def next_section(self) -> None:
        """Displays the next section in the textbox

        Returns: None
        """
        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to display. Load a file first.")
            return

        if self.current_section + 1 == len(self.sections):
            self._load_sections()

        if self.current_section + 1 == len(self.sections):
            messagebox.showinfo("End of File", "No more sections to display.")

        # Update the current subject if we have moved to a new one
        if self.current_subject not in self.sections[self.current_section]:
            self.current_subject = self.subs.search(self.sections[self.current_section]).group(0).replace("(", "").replace(")", "")

        self._update_section()
        self.current_section += 1
        self.last_viewed_section = max(self.last_viewed_section, self.current_section)
        self._show_section()

    def previous_section(self) -> None  :
        """
        Displays the previous section in the textbox

        Returns: None
        """
        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to display. Load a file first.")
            return

        if self.current_section == 0:
            messagebox.showinfo("No previous section", "The current section is the first loaded section. "
                                                       "To go back further, save and jump to a previous chunk.")
            return

        self._update_section()
        self.last_viewed_section = max(self.last_viewed_section, self.current_section)
        self.current_section -= 1
        self._show_section()
    
    def delete_section(self) -> None:
        """ Delete the current section.
        """
        # Failsafe to make sure a file is loaded
        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to display. Load a file first.")
            return
        # Update section parameters
        self._update_section()
        self.last_viewed_section = max(self.last_viewed_section, self.current_section)
        # Delete the current section
        self.sections.pop(self.current_section)
        # If the current section is the last section, move back one section instead of forward
        if self.current_section == len(self.sections):
            self.current_section -= 1
        self._show_section()
    
    def next_subject(self) -> None:
        """ Move to the next subject.
        """
        # Failsafe to make sure a file is loaded
        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to display. Load a file first.")
            return
        # Update section parameters
        self._update_section()
        self.last_viewed_section = max(self.last_viewed_section, self.current_section)
        # Find the next subject
        for j in range(self.current_chunk, len(self.chunks)):
            self.current_chunk = j
            self._load_sections()
            if self._find_subject():
                if all((c != j and s != self.current_section) for s, c in self.subject_idx):
                    self.subject_idx.append((self.current_section, j))
                self.current_sub_idx += 1
                break
        self._show_section()

    def jump_chunk(self) -> None:
        """
        Loads sections from a desired chunk

        Returns: None
        """
        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to display. Load a file first.")
            return

        # Show warning dialog before loading a specific chunk
        if messagebox.askokcancel("Warning", "Loading a specific chunk will overwrite any unsaved progress. "
                                             "Do you want to continue?"):

            # check input
            try:
                chunk_num = int(self.chunk_entry.get())
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid integer for chunk number.")
                return

            if chunk_num >= len(self.chunks):
                messagebox.showwarning("Invalid Chunk", "Chunk number out of range.")
                return

            self.current_chunk = chunk_num
            self.current_section = 0
            self.last_viewed_section = 0

            self.sections = []
            searching = True
            start_chunk = self.current_chunk
            while searching:
                searching = False
                start_chunk = self.current_chunk
                try:
                    self._load_sections()  # Load sections from the specific chunk
                except ValueError:
                    searching = True
                    self.current_chunk += 1
                    if chunk_num >= len(self.chunks):
                        messagebox.showwarning("Invalid Chunk", "Chunk number out of range.")
                        return

            self.start_chunk = start_chunk
            self._show_section()

    def detect_type(self) -> None:
        """
        If no type parameter is present, remove the first line and set it as the type parameter

        Returns: None
        """
        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to display. Load a file first.")
            return

        self._update_section()
        lines = self.sections[self.current_section].split("\n")
        result = re.search(r" T ", lines[0])

        # If type is already present, move on
        if result is None:
            return

        # If type is not present, use the first non empty line to determine the type
        line_index = 1
        while line_index < len(lines):
            if question_type := lines[line_index].strip():
                break
            else:
                line_index += 1

        if line_index == len(lines):
            messagebox.showinfo("No type detected", "Sections seems to be empty, so no type could be extracted")
            return

        if question_type[0] == "[" or question_type[0] == "(" or question_type[0] == "{":
            question_type = question_type[1:-1]

        if len(question_type) > 40:
            question_type = question_type[:40]

        # Replace the type in the header
        lines[0] = f"{lines[0][:result.span()[0]]} ({question_type}) {lines[0][result.span()[1]:]}"

        # Remove the first line from the section
        del lines[line_index]

        self.sections[self.current_section] = "\n".join(lines)
        self._show_section()

    def to_latex(self) -> None:
        """Detects lists and formats them according to latex

        Returns: None
        """

        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to display. Load a file first.")
            return

        self._update_section()
        text = self.sections[self.current_section]
        header = text.split("\n")[0]
        body = text[len(header):]

        # Pattern to detect numbered lists (e.g., 1. Text 2. Text ...)
        list_pattern = r'(?:\d+\.?\s)([^\d]+)'

        # Function to convert the detected lists to a single LaTeX list
        def list_to_latex(match):
            items = re.findall(list_pattern, match.group(), re.DOTALL)
            latex_list = "\\begin{enumerate}\n"
            for item in items:
                latex_list += f"\\item {item.strip()}\n"
            latex_list += "\\end{enumerate}"
            return latex_list

        # Replace all detected lists with their LaTeX versions
        new_body = re.sub(r'((?:\d+\.?\s+.+?)(?=(?:\d+\.?\s+)|\n\n|$))+', list_to_latex, body, flags=re.DOTALL)

        new_body = re.sub(r'Answers', r"\\section{Answers}", new_body)
        self.sections[self.current_section] = header + new_body
        self._show_section()

    def remove_brackets(self) -> None:
        """Removes all square and wavy brackets from the current section

        Returns: None
        """
        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to display. Load a file first.")
            return

        self._update_section()
        text = self.sections[self.current_section]
        header = text.split("\n")[0]
        body = text[len(header):]
        body = re.sub(r"[(\[\]{}]", "", body)
        self.sections[self.current_section] = header + body
        self._show_section()

    def scale_font_size(self, factor: float) -> None:
        self.textbox_font.config(size=int(self.textbox_font["size"] * factor))

    def _load_sections(self) -> None:
        """
        Loads all sections from the current chunk and move to next chuck.

        Sections are detected by subject names (SUBJECTS). The last section
        might contain contents of next chuck.

        Returns: None
        """
        # Find the start of the first section, there should always be one
        result = self.subs.search(self.chunks[self.current_chunk])
        if result is None:
            raise ValueError(f"Chunk {self.current_chunk} has no beginning question")

        section_start = result.span()[0]
        # Find the end of the first section
        result = self.subs.search(self.chunks[self.current_chunk],
                                  pos=result.span()[1])

        # while there is a beginning of a new section, add old section and repeat
        while result is not None:
            section_end = result.span()[0]
            self.sections.append(self.chunks[self.current_chunk][section_start:section_end])
            section_start = section_end
            result = self.subs.search(self.chunks[self.current_chunk],
                                      pos=result.span()[1])

        # look at the next chunk to get the last section
        text = self.chunks[self.current_chunk][section_start:]
        if self.current_chunk + 1 < len(self.chunks):
            self.current_chunk += 1
            result = self.subs.search(self.chunks[self.current_chunk])
            # In case there is no new section in the next chunk, continue to next chunk
            while result is None and self.current_chunk < len(self.chunks):
                text += self.chunks[self.current_chunk]
                self.current_chunk += 1
                result = self.subs.search(self.chunks[self.current_chunk])
            # If there is a new section in the next chunk, add the text up to that point
            if result is not None:
                section_end = result.span()[0]
                text += self.chunks[self.current_chunk][:section_end]
        self.sections.append(text)

    def _show_section(self) -> None:
        """
        Load current section into textbox

        Returns:
        """
        if self.current_section < len(self.sections):
            self.textbox.delete(1.0, tk.END)
            self.textbox.insert(tk.END, self.sections[self.current_section])
            self._update_chunk_label()

    def _update_chunk_label(self) -> None:
        """
        Set the chunk label to current chunk

        Returns: None
        """
        self.chunk_entry.delete(0, tk.END)
        self.chunk_entry.insert(1, str(self.current_chunk - 1))

    def _update_section(self) -> None:
        """
        Replace local contents of the current section with contents of the textbox

        Returns: None
        """
        self.sections[self.current_section] = self.textbox.get(1.0, tk.END).strip()

    def _find_subject(self) -> bool:
        """ Find the next subject in the sections list.

        Returns:
            bool: True if a new subject was found in this section, False otherwise.
        """
        for i in range(self.current_section + 1, len(self.sections)):
            subj = self.subs.search(self.sections[i]).group(0).replace("(", "").replace(")", "")
            if  subj != self.current_subject and "(" + subj + ")" in self.sections[i] and " D " in self.sections[i]:
                self.current_section = i
                self.current_subject = subj
                return True
        return False

    def apply_bold(self) -> None:
        """Apply LaTeX bold formatting to the selected text."""
        self._apply_latex_format("\\textbf{", "}")

    def apply_italic(self) -> None:
        """Apply LaTeX italic formatting to the selected text."""
        self._apply_latex_format("\\textit{", "}")

    def apply_underline(self) -> None:
        """Apply LaTeX underline formatting to the selected text."""
        self._apply_latex_format("\\underline{", "}")

    def _apply_latex_format(self, prefix: str, suffix: str) -> None:
        """Apply a given LaTeX format (prefix and suffix) to the selected text."""
        try:
            selected_text = self.textbox.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.textbox.delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.textbox.insert(tk.INSERT, f"{prefix}{selected_text}{suffix}")
        except tk.TclError:
            messagebox.showwarning("No Selection", "Please select text to format.")


if __name__ == "__main__":
    root = tk.Tk()
    if len(sys.argv) == 1:
        app = TextEditor(root)
    elif len(sys.argv) == 3:
        app = TextEditor(root, sys.argv[1], sys.argv[2])
    else:
        raise ValueError("Invalid number of arguments. Please provide either no arguments or two arguments (filepath_in, filepath_out).")
    root.mainloop()
