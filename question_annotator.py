# Copyright (C) 2024  Edion Management Systems
import tkinter as tk
from tkinter import filedialog, messagebox
import re

# TODO:     1. Implement config for chunk size and regex
#           2. Automated header (+ footer?) annotation

SEP = "\n\n\n"

SUBJECTS = ["Computer Science", "Science", "Mathematics", "Social Studies", "History", "Geography",
            "Spanish", "French", "German", "Dutch", "English"]


class TextEditor:
    """
    A simple text editor for annotating questions in the specified format
    """

    def __init__(self, root: tk.Tk):
        """
        Initialize the TextEditor class.

        Args:
            root: Tkinter root object
        """
        self.root = root
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

        self.type_button = tk.Button(self.button_frame, text="Previous Subject", command=self.previous_subject)
        self.type_button.pack(side='left')

        self.type_button = tk.Button(self.button_frame, text="Next Subject", command=self.next_subject)
        self.type_button.pack(side='left')

        self.jump_chunk = tk.Button(self.button_frame, text="Jump To Chunk", command=self.jump_chunk)
        self.jump_chunk.pack(side='right')

        self.chunk_number_label = tk.Label(self.button_frame, text="/ -")
        self.chunk_number_label.pack(side='right')

        self.chunk_entry = tk.Entry(self.button_frame, width=8)
        self.chunk_entry.pack(side='right')

        # Add chunk and section selectors
        self.chunk_label = tk.Label(self.button_frame, text="Chunk:")
        self.chunk_label.pack(side='right')

        # Initialize variables
        self.filepath_old = None
        self.file_old = None
        self.filepath_new = None
        self.file_new = None
        self.chunk_size = 100  # lines per chunk
        self.start_chunk = 0
        self.current_chunk = 0
        self.chunks = []
        self.current_subject = "Computer Science"
        self.subject_idx = [(0, 0)]
        self.current_sub_idx = 0
        self.current_section = 0
        self.last_viewed_section = 0
        self.sections = []
        self.subs = re.compile(r"\(?(" + "|".join(SUBJECTS) + ")")

        self.root.bind("<F1>", lambda _: self.load_files())
        self.root.bind("<F2>", lambda _: self.save_sections())
        self.root.bind("<F3>", lambda _: self.previous_section())
        self.root.bind("<F4>", lambda _: self.next_section())
        self.root.bind("<F5>", lambda _: self.detect_type())
        self.root.bind("<F6>", lambda _: self.delete_section())

    def load_files(self):
        """
        Load a text file to read from and a text file to write to.

        Returns:

        """
        if self.chunks and not messagebox.askokcancel("Warning", "Loading new files will overwrite any unsaved progress. "
                                                                 "Do you want to continue?"):
            return

        self.filepath_old = filedialog.askopenfilename(title="Select a Text File to read from",
                                                       filetypes=(("Text Files", "*.txt"),))
        if not self.filepath_old:
            return

        self.filepath_new = filedialog.askopenfilename(title="Select a Text File to write to",
                                                       filetypes=(("Text Files", "*.txt"),))
        if not self.filepath_new:
            return

        with open(self.filepath_old, 'r', encoding="utf8") as file:
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

    def save_sections(self):
        """
        Saves loaded sections to file

        Returns: None
        """

        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to save. Load a file first.")
            return

        self._update_section()

        with open(self.filepath_new, 'w', encoding="utf8") as file:
            for section in self.sections[:self.last_viewed_section+1]:
                while section and section[-1] == "\n":
                    section = section[:-1]
                file.write(section + f"\n\n\n")

        messagebox.showinfo("Saved Sections", f"Saved {self.last_viewed_section + 1} sections in chunks "
                                              f"{self.start_chunk} to {self.current_chunk}")

    def next_section(self):
        """
        Displays the next section in the textbox

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

    def previous_section(self):
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
    
    def delete_section(self):
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
    
    def next_subject(self):
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

    def previous_subject(self):
        """ Move to the previous subject.
        """
        # Failsafe to make sure a file is loaded
        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to display. Load a file first.")
            return
        # Update section parameters
        self._update_section()
        self.last_viewed_section = max(self.last_viewed_section, self.current_section)
        self.current_sub_idx = max(0, self.current_sub_idx - 1)
        self.current_chunk = self.subject_idx[self.current_sub_idx][1]
        self.current_section = self.subject_idx[self.current_sub_idx][0]
        self.current_subject = self.subs.search(self.sections[self.current_section]).group(0).replace("(", "").replace(")", "")
        self._show_section()

    def jump_chunk(self):
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

    def detect_type(self):
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

        # If type is not present, use the first line to determine the type
        question_type = lines[1] if lines[1] else lines[2]

        while question_type[-1] == " ":
            question_type = question_type[:-1]

        if question_type[0] == "[" or question_type[0] == "(" or question_type[0] == "{":
            question_type = question_type[1:-1]

        if len(question_type) > 40:
            question_type = question_type[:40]

        # Replace the type in the header
        lines[0] = f"{lines[0][:result.span()[0]]} ({question_type}) {lines[0][result.span()[1]:]}"

        # Remove the first line from the section
        if lines[1].replace(" ", "") == "":
            del lines[2]
        else:
            del lines[1]

        self.sections[self.current_section] = "\n".join(lines)
        self._show_section()

    def _load_sections(self):
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

    def _show_section(self):
        """
        Load current section into textbox

        Returns:
        """
        if self.current_section < len(self.sections):
            self.textbox.delete(1.0, tk.END)
            self.textbox.insert(tk.END, self.sections[self.current_section])
            self._update_chunk_label()

    def _update_chunk_label(self):
        """
        Set the chunk label to current chunk

        Returns: None
        """
        self.chunk_entry.delete(0, tk.END)
        if self.current_chunk != 0:
            self.chunk_entry.insert(1, str(self.current_chunk - 1))

    def _update_section(self):
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
                
if __name__ == "__main__":
    root = tk.Tk()
    app = TextEditor(root)
    root.mainloop()
