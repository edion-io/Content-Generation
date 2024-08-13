# Content-Generation
Official Edion repository implementing models and tools used for educational content-generation.

TO BE MORE DETAILED

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



## Moduls

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

You can also load a different section of the (quite long) questions file. To do so, enter the desired chunk in the 
respective field (bottom right) and press the button. This action overwrites all local changes so make sure to save if
necessary. If everything worked, you should now be able to edit the data from a different part of the file. Saving the 
processed data works normally, note that only edited (technically viewed) sections will be saved, the preceding chunks 
will be ignored.

**Technical details**:
Upon loading, the program reads the entire file. This is not the most efficient but should suffice for now. The text
is read and stored in batches (chunks) of a predetermined size (use 100 lines). This should allow the 
program to be extended if need be. 

The program loads sections based on chunks, i.e. all sections within a chunk are preloaded once the first section is 
needed. When questions span across two chunks, the program looks ahead to the next chunk to complete the question.

The segmentation into sections uses regex and tries to capture the beginning of a question by the subject names. This 
method of segmentation is suboptimal as the names can also occur in the question text. Therefore, we use triple newlines
to separate the sections.