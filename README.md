# Content-Generation
Official Edion repository implementing models and tools used for educational content-generation.

## Data
The data is stored in a single file, `questions.txt`. The file contains a large number of questions, each separated by
three newlines. The questions are further divided into sections based on the subject they belong to. The sections are
separated by the name of the subject enclosed in parentheses. The file is roughly composed as follows:


| Subject          | Number of questions | Range (lines) |
|------------------|---------------------|---------------|
| Computer Science | 500                 | 0-3200        |
| English          | 1000                | 3200-11500    |
| Geography        | 1000                | 11500-17700   |
| History          | 450                 | 17700-21200   |
| Social Studies   | 666                 | 21200-25800   |
| Dutch            | 200                 | 25800-30600   |
| French           | 240                 | 30600-32100   |
| Spanish          | 230                 | 32100-40800   |
| German           | 100                 | 40800-43900   |
| Science          | 600                 | 43900-50000   |
| Maths            | 5160                | 50000-89900   |



## Modules

### Extractor
The extractor module can be used to extract questions from images or to refine raw text questions into usable questions. It functions by first extracting the relevant images or raw questions from text, and then either using the Batch GPT4-o mini API as an advanced OCR tool or a means to format questions automatically. 

**Working with images**
You can extract relevant images from PDF files (textbooks) before sending them over to the API. This is done using:
```
python extractor.py e [-sp] <start_page> <end_page>
```
Where `start_page` and `end_page` are the range of pages you want to extract images from. Make sure you specify the folder containing the relevant textbooks in the parameter `PATH`. The images are saved in the folder specified under the parameter `IMAGE_FOLDER`. Running this command will both extract the images, upload them to cloudinary and compile them into batched chat-completion requests, stored in the folder specified by `BATCH_FOLDER`. If for some reason you want to skip the image extraction step (you already have images), you can use the optional `-sp` flag.

As the API will perform the OCR on the images, you must provide a valid prompt for the API to use, that instructs it on how you would like questions to be extracted. This can be done by specifying the prompt in the `PROMPT` parameter. It is mandatory to separate questions with a `*NEW*` string (see below). It is also imperative to test the prompt on the API before running the extractor as this costs money. For images, it is advised to use the actual API playground to test the prompt as the ChatGPT version of GPT4o-mini does not have access to file uploads.

**Working with text**
You can extract questions from text files using:
```
python extractor.py et
```
Make sure you specify the file containing the raw questions in the parameter `PATH`. Raw questions are expected to be separated by their respective question header, which we specify under the parameter `HEADER`. For example, for a .txt of German questions that have answers,`HEADER` might be set to `German T D G (With Answer)`. 

Similarly as with images, the `PROMPT` parameter will guide the method used by the API to further process the text. The prompt should be tested on the API before running the extractor. Since you are only dealing with text, you may use either the playground or the ChatGPT version of GPT4o-mini.

**Working with questions**
You can annotate questions from the question file using:
```
python extractor.py q <subject>
```
Where `subject` is the subject you want to annotate. This will take all the questions you specify in the file constant `PATH` and compile them into a batched chat-completion request, stored in the folder specified by `BATCH_FOLDER`. The prompt also needs to be scpeified in the `PROMPT` parameter.

**Submitting a batch**
Once you have extracted the images or raw questions, you can submit a specific batch to the API using:
```
python extractor.py sb <batch_file_name>
```
or submit all batches in the  using:
```
python extractor.py ab
```
Note that for `sb` you do not need to specify the file extension, as the program automatically assumes you are using `.jsonl` files. With a batch submitted, you can then check the status of the batch using:
```
python extractor.py s <batch_job_id>
```
where `batch_job_id` is the id of the batch job you want to check. This can be found in the batch_job_id.txt file of this repository. Note that since we are using the Batch API, the job can theoretically take up to 24h to complete (though it usually takes less than 2 minutes).

**Retrieving the results**
Once the batch job is completed, you can retrieve all of the results using:
```
python extractor.py r [-t | -p]
```
The `-t` flag will retrieve the text results, while the `-p` flag will retrieve the image results. For text results, questions are automatically extracted, while for image results, it is assumed that the returned questions will be separated by the `*NEW*` string. In either cases, results are stored in the `questions.txt` file. 

### Question Annotator
**Description**: 
Python script to facilitate manual annotation/quality control of extracted questions

**Instructions**:
After starting the program, first load files. You are asked to load two files: first one to read from
(this is questions.txt, the raw unprocessed data), secondly one to write to (this is a different file to store the
processed output). The output file will overwrite previous content so watch out.

After loading the files, if everything went well you should see the first question in the editor window. Now you can 
make changes, add or delete parts. Once you are happy, you can move to the next section (question), your changes are 
automatically stored temporarily when you move forth (or back).

Once you are done (or just feel like it) you can save the sections that you edited. This will overwrite the contents of
your previously selected output file with the sections you edited.

You can also load a different chunk of the (quite long) questions file. To do so, enter the desired chunk in the 
respective field (bottom right) and press Enter. This action overwrites all local changes so make sure to save if
necessary. If everything worked, you should now be able to edit the data from a different part of the file. Saving the 
processed data works normally, note that only edited (technically viewed) sections will be saved, the preceding chunks 
will be ignored.

**Functionality**:

- `Ctrl + O` or `F1`: Load files, first the file to read from, then the file to write to. The output file will be 
  overwritten. This action also resets the editor and deletes all local changes.
- `Ctrl + S` or `F2`: Save sections that were edited (viewed). Only the edited sections will be saved, the preceding 
  chunks will be ignored. The output file will be overwritten.
- `Ctrl + Alt + Left` or `F3`: Go to the previous section. If the current section was edited, the changes will be saved.
- `Ctrl + Alt + Right` or `F4`: Go to the next section. If the current section was edited, the changes will be saved.
- `Ctrl + T` or `F5`: Detect type. This is an experimental feature that will replace an empty type parameter (T) with 
  the first line after the question header that is not empty.
- `F6`: Delete current section (experimental)
- `Ctrl + L` or `F7`: Convert to LaTeX. This will convert the current section to LaTeX format by wrapping Section
  headers **Solutions** and **Hints** in LaTeX commands, converting Markdown text formatting to latex
  (single * to italic, double ** to bold, and underscore _ to underline), and replacing potentially multiple numerical 
  lists with LaTeX lists (1 dog 2 cat -> \begin{enumerate} \item dog \item cat \end{enumerate}).
  The list replacement is not perfect and will not work if there are numbers that are not part of denoting lists. In 
  those cases, use the selected list replacement feature (Ctrl + Alt + L).
- `Ctrl + Alt + L`: Format selected list. This works a bit like the list replacement in the LaTeX conversion but
  more fine-grained. It will convert the selected lines to a LaTeX list. It only works for lists that have each item on
  a new line, however it accepts both alphabetical and numerical lists and detects and removes punctuation.
- `Ctrl + R`: Replace all instances of a string with a different string. If text is selected, the selection will
  automatically be used as the term to be replaced.
- `Ctrl + Alt + R`: Apply the last replacement action again. This is useful if you want to replace the same term
  in multiple questions in case of repeating patterns.
- `F8`: Remove brackets. Kind of a stupid feature that removes all brackets from the current section.
- `Ctrl + Z` and `Ctrl + Y`: Undo and redo. This is a bit buggy as it only works for changes to the text box contents
  and not for the local changes like moving between sections. Custom functions like LaTeX conversion can be undone,
  but will require two undo actions, with the first one removing all text in the editor.
- `Ctrl + B`: Apply bold
- `Ctrl + I`: Apply italic
- `Ctrl + U`: Apply underline
- `Ctrl + +`: Increase font size
- `Ctrl + -`: Decrease font size


**Technical details**:
Upon loading, the program reads the entire file. This is not the most efficient but should suffice for now. The text
is read and stored in batches (chunks) of a predetermined size (use 100 lines). This should allow the 
program to be extended if need be. Chunk numbers start at 0.

The program loads sections based on chunks, i.e. all sections within a chunk are preloaded once the first section is 
needed. When questions span across two chunks, the program looks ahead to the next chunk to complete the question.

The segmentation into sections uses regex and tries to capture the beginning of a question by the subject names. This 
method of segmentation is suboptimal as the names can also occur in the question text. Therefore, we use triple newlines
to separate the sections.

Section numbers start at 1 and are incremented by 1 for each new section. The program keeps track of the current
section number and the current chunk number. The section number is used to identify the sections in the output file.
Note that section numbers are not global, they depend on what chunk was loaded first. E.g. if you load chunk 0 and then
jump to chunk 1, the section numbers will start at 1 again.