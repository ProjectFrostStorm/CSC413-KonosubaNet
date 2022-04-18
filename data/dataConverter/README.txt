This is a custom written tool powered by vanilla Javascript for the purpose of processing and analyzing data for the 2022 CSC413 Final Project: "KonosubaNet". 
This tool is licensed under the same as the entire project.

Tool written by (and for) the member in charged of data processing: Allen Yao

Instructions on how to use the tool can be found on the tool interface itself. A copy of the instructions can be found below.

Installation Instructions:
- Download the <index.html> and <conversion.js> files and ensure they are in the same directory.
- Open the <index.html> file with any modern browser. Google Chrome (Version 100.0.4896.127) was the browser used during development and usage.
The instructions and interface can be found on the main page of the tool. If the instructions were unreadable for any reason, a copy can be found below:


_____________________________________________________________________________________________________
Note: Some information is outputted to the Javascript console during runtime of the tool, which can assist with problems or serve as a loading bar if things are taking too long.
It is recommended that the page is refreshed between file processes. Sometimes, repeated usage can cause the page to be stuck, which can be fixed with a refresh.

<h3>Data Preparation Instructions</h3>

Ensure input data is correctly formatted in plaintext (with UTF-8 encoding). This requires line breaks separating every distinct line (multiple line breaks in a row will 
be ignored by the tool and so can be used to increase readability) and proper volume, chapter, and part delimiters. Those delimiters are the following strings on their 
own lines: 
<pre>^VOL SPLIT^</pre>
<pre>^CHAPTER SPLIT^</pre>
<pre>^PART SPLIT^</pre>
An example of a valid format can be found under the "examples" directory.

An optional tagging system is available; it can be used in the Processing and Analysis functions. To use this tagging system, ensure all lines are formatted correctly:

All dialogue must begin with a double quotation mark (or begin with an opening angle bracket immediately followed by a double quotation in the case of 
Aegis' dialogue).

<p>All translation notes must begin with an opening angle bracket (and not immediately followed by a double quotation).</p>

<p>All quotations (lines that are narration lines nor dialogue) must begin with a single quotation mark.</p>

<p>All remaining lines will be classified with narration.</p>

Use the file uploader ("Choose File" button) and upload input data.

Use the "Plaintex to JSON" button to process file into machine processable JSON. This will download a JSON file to your machine with the name <i>data.json</i>.
Note that the conversion process will turn directioned single and double quotation to non-directed single and double quotation. It will also remove the carriage return
character added by Windows (the \r character inserted before every \n).

<h3>Data Processing Instructions</h3>

The data processing function only takes files in a JSON format (with the .json extension). There is no format validation, so ensure files used for Data Processing 
were returned by the Data Preparation function. Use the file uploader ("Choose File" button) to upload prepared data file.

Use the Filtering checkboxes in the "JSON to Data Options" to filter out lines of a certain tag. In the final data used for the project, translator notes are filtered 
out.

Use the "JSON to Data" button to process JSON file into final data format. This will download a TXT file to your machine with the name <i>data.txt</i>.
The final data format separates lines with consist single line breaks (\n) and parts with two line breaks (\n\n). This data file can be used by the network for training.

<h3>Data Analysis Instructions</h3>

The data analysis function only takes files in a JSON format (with the .json extension). There is no format validation, so ensure files used for Data Processing 
were returned by the Data Preparation function. Use the file uploader ("Choose File" button) to upload prepared data file.

The following options are available for data analysis under the "JSON Data Analysis Options":

<p>Use the Filtering checkboxes in the "JSON to Data Options" to filter out lines of a certain tag. </p>

Use the "Output Analysis to File" checkbox to determine where the analysis output will go. If unchecked, it will be outputted to the HTML page itself, under 
the "HTML Output" header. Otherwise, it will download a TXT file to your machine with the name <i>stats.txt</i> containing the output of the analysis.

Use the "Output Character/Word Distribution to CSV" checkboxes to download a CSV file to your machine with the name <i>dist.csv</i> containg the occurrences of 
the characters or words in the data file. The CSV file will be formatted with ("&lt;char/word>,&lt;number>") pairs per line. It is not recommended to check both 
boxes at once, as that will lump both character and word occurrences into a single CSV file.

Use the "Analyse JSON" button to return the analysis of the data file. Any downloaded files will be triggered now (the browser may prompt to allow multiple files
to be downloaded at once).