import tkinter as tk
from tkinter import filedialog, messagebox
import re

question_tags = ["Computer Science", "Science", "Math", "Spanish Language", "French", "German", "Dutch",
                     "Social Studies", "English"]

class TextEditor:

    def __init__(self, root):
        self.root = root
        self.root.title("Text Section Editor")

        # Create widgets
        self.textbox = tk.Text(self.root, wrap='word')
        self.textbox.pack(expand=1, fill='both')

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(fill='x')

        self.load_button = tk.Button(self.button_frame, text="Load File", command=self.load_file)
        self.load_button.pack(side='left')

        self.save_button = tk.Button(self.button_frame, text="Save Section", command=self.save_section)
        self.save_button.pack(side='left')

        self.next_button = tk.Button(self.button_frame, text="Next Section", command=self.next_section)
        self.next_button.pack(side='right')

        # Initialize variables
        self.filepath_old = None
        self.file_old = None
        self.filepath_new = None
        self.file_new = None
        self.chunk_size = 1000  # Characters per chunk
        self.current_chunk = 0
        self.chunks = []
        self.current_section = 0
        self.sections = []
        self.subs = re.compile("|".join(question_tags))

    def load_file(self):
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
            while True:
                text_chunk = file.read(self.chunk_size)
                if not text_chunk:
                    break
                self.chunks.append(text_chunk)

        self.current_chunk = 0
        self.current_section = 0
        self.load_sections()
        self.show_section()

    def load_sections(self):
        section_start = 0
        section_end = 0

        # Find the start of the first section, there should always be one
        result = self.subs.search(self.chunks[self.current_chunk])
        if result is None:
            messagebox.showwarning("Runaway Section", "Cannot find new section start")
        else:
            section_start = result.span()[0]

        # Find the end of the first section
        result = self.subs.search(self.chunks[self.current_chunk],
                                  pos=result.span()[1])
        # while there is a beginning of a new section, add old section and repeat
        while result is not None:
            section_end = result.span()[0]
            self.sections.append(self.chunks[self.current_chunk][section_start:section_end])
            print(f"adding section {len(self.sections)}, [{section_start}:{section_end}]")
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
        print(f"adding section {len(self.sections)}, last question of chunk")
        print(f"current chunk: {self.current_chunk}, current section: {self.current_section}")

    def show_section(self):
        if self.current_section < len(self.sections):
            self.textbox.delete(1.0, tk.END)
            self.textbox.insert(tk.END, self.sections[self.current_section])

    def save_section(self):
        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to save. Load a file first.")
            return

        self.sections[self.current_section] = self.textbox.get(1.0, tk.END).strip()
        with open(self.filepath_new, 'w', encoding="utf8") as file:
            for section in self.sections:
                file.write(section + "\n")

    def next_section(self):
        if not self.chunks:
            messagebox.showwarning("No Sections", "No sections to display. Load a file first.")
            return

        if self.current_section + 1 == len(self.sections):
            self.load_sections()

        self.current_section += 1

        if self.current_section == len(self.sections):
            messagebox.showinfo("End of File", "No more sections to display.")

        self.show_section()

        print(f"current chunk: {self.current_chunk}, total chunks: {len(self.chunks)}, \n"
              f"current section: {self.current_section} total sections: {len(self.sections)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextEditor(root)
    root.mainloop()