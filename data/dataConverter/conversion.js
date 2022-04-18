
let x = "";
let printJSONResult = false;


function convertToJSON()
{
    //
    let file = document.getElementById("fileUp").files[0];
    
    //var myUploadedFile = document.getElementById("myFile").files[0];
    var fileContent;

    //https://stackoverflow.com/questions/750032/reading-file-contents-on-the-client-side-in-javascript-in-various-browsers
    var reader = new FileReader();
    reader.readAsText(file, "UTF-8");
    reader.onload = function (evt) {
        fileContent = evt.target.result;
        //document.getElementById("fileContents").innerHTML = evt.target.result;
        //document.getElementById("fileContents").innerHTML = fileContent;

        returnJSON(fileContent);

        //alert(fileContent);
        //console.log(fileContent);
    }
    reader.onerror = function (evt) {
        document.getElementById("fileContents").innerHTML = "error reading file";
    }
}

function convertToData()
{
    //
    let file = document.getElementById("fileUp").files[0];
    
    //var myUploadedFile = document.getElementById("myFile").files[0];
    var fileContent;

    //https://stackoverflow.com/questions/750032/reading-file-contents-on-the-client-side-in-javascript-in-various-browsers
    var reader = new FileReader();
    reader.readAsText(file, "UTF-8");
    reader.onload = function (evt) {
        fileContent = evt.target.result;
        //document.getElementById("fileContents").innerHTML = evt.target.result;
        //document.getElementById("fileContents").innerHTML = fileContent;

        returnData(fileContent);

        //alert(fileContent);
        //console.log(fileContent);
    }
    reader.onerror = function (evt) {
        document.getElementById("fileContents").innerHTML = "error reading file";
    }
}

function analyseJSON()
{
    //
    let file = document.getElementById("fileUp").files[0];
    
    //var myUploadedFile = document.getElementById("myFile").files[0];
    var fileContent;

    //https://stackoverflow.com/questions/750032/reading-file-contents-on-the-client-side-in-javascript-in-various-browsers
    var reader = new FileReader();
    reader.readAsText(file, "UTF-8");
    reader.onload = function (evt) {
        fileContent = evt.target.result;
        //document.getElementById("fileContents").innerHTML = evt.target.result;
        //document.getElementById("fileContents").innerHTML = fileContent;

        returnAnalysis(fileContent);

        //alert(fileContent);
        //console.log(fileContent);
    }
    reader.onerror = function (evt) {
        document.getElementById("fileContents").innerHTML = "error reading file";
    }
}

function returnData(jsonFile)
{
    let text = JSON.parse(jsonFile);
    let output = "";

    let maskNarration = document.getElementById("narrationMask").checked;
    let maskDialogue = document.getElementById("dialogueMask").checked;
    let maskTLNote = document.getElementById("tlnoteMask").checked;
    let maskQuote = document.getElementById("quoteMask").checked;

    let lastLineWasLineBreak = false; //If all lines in a part are filtered out, don't add that part's line break in

    for(let vol = 0; vol < text.volumes.length; vol++)
    {
        for(let chapter = 0; chapter < text.volumes[vol].chapters.length; chapter++)
        {
            for(let part = 0; part < text.volumes[vol].chapters[chapter].parts.length; part++)
            {
                for(let sentence = 0; sentence < text.volumes[vol].chapters[chapter].parts[part].sentences.length; sentence++)
                {
                    /*
                    let isNarration = text.volumes[vol].chapters[chapter].parts[part].sentences[sentence].isNarration;
                    let isDialogue = text.volumes[vol].chapters[chapter].parts[part].sentences[sentence].isDialogue;
                    let isTLNote = text.volumes[vol].chapters[chapter].parts[part].sentences[sentence].isTLNote;
                    let isQuote = text.volumes[vol].chapters[chapter].parts[part].sentences[sentence].isQuote;
                    */

                    let type = text.volumes[vol].chapters[chapter].parts[part].sentences[sentence].type;

                    //Filter out specified sentences
                    if(maskNarration && type == "narration")
                    {
                        console.log("Skipping narration sentence: " + sentence + " in part: " + part + " in chapter: " + chapter + " in vol: " + vol);
                        continue;
                    }
                    else if(maskDialogue && type == "dialogue")
                    {
                        console.log("Skipping dialogue sentence: " + sentence + " in part: " + part + " in chapter: " + chapter + " in vol: " + vol);
                        continue;
                    }
                    else if(maskTLNote && type == "tlnote")
                    {
                        console.log("Skipping tlnote sentence: " + sentence + " in part: " + part + " in chapter: " + chapter + " in vol: " + vol);
                        continue;
                    }
                    else if(maskQuote && type == "quote")
                    {
                        console.log("Skipping quote sentence: " + sentence + " in part: " + part + " in chapter: " + chapter + " in vol: " + vol);
                        continue;
                    }

                    if(lastLineWasLineBreak)
                    {
                        lastLineWasLineBreak = false; //We're adding in a line of text now
                    }

                    output += text.volumes[vol].chapters[chapter].parts[part].sentences[sentence].text + "\n";

                    console.log("Processing " + type + " sentence: " + sentence + " in part: " + part + " in chapter: " + chapter + " in vol: " + vol);
                }

                //Add the second line break delimiting parts
                if(!lastLineWasLineBreak)
                {
                    output += "\n";
                    lastLineWasLineBreak = true;
                }
            }
        }
    }

    //https://stackoverflow.com/questions/19721439/download-json-object-as-a-file-from-browser
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(output);
    var dlAnchorElem = document.getElementById('downloadAnchorElem');
    dlAnchorElem.setAttribute("href",     dataStr     );
    dlAnchorElem.setAttribute("download", "data.txt");
    dlAnchorElem.click();
}

function returnJSON(fileContent)
{
    console.log(fileContent.length);

    /*
    for(let x = 0; x < fileContent.length; x++)
    {
        document.getElementById("fileContents").innerHTML += fileContent[x];
        if(fileContent[x] == "\n")
        {
            document.getElementById("fileContents").innerHTML += " EOL" + "<br />";
        }
    }
    */

    let sentences = [];
    let partSplit = []; //A new part starts at this sentence index
    let chapterSplit = [];
    let volSplit = [];
    let numSentences = 0;
    let sentenceStart = 0;
    for(let x = 0; x < fileContent.length; x++)
    {
        //Sentence break
        if(fileContent[x] == "\n")
        {
            //Two linebreaks are back to back
            if(sentenceStart == x - 1)
            {
                sentenceStart = x;
                continue;
            }

            let nextSentence = fileContent.slice(sentenceStart, x);
            if((nextSentence.replace("\n","")).replace("\r", "") == "") //Skip empty sentence
            {
                sentenceStart = x + 1;
                continue;
            }
            sentences.push(fileContent.slice(sentenceStart, x));
            sentenceStart = x + 1;
            numSentences++;
        }
        
        /*
        //Part break
        if(fileContent.length - x < 13 && fileContent.slice(x, x + 13) == "^PART SPLIT^\n")
        {
            sentenceStart = x + 13;
            partSplit.push(numSentences);
        }
        
        //Chapter break
        if(fileContent.length - x < 16 && fileContent.slice(x, x + 16) == "^CHAPTER SPLIT^\n")
        {
            sentenceStart = x + 16;
            chapterSplit.push(numSentences);
        }
        
        //Volume break
        if(fileContent.length - x < 12 && fileContent.slice(x, x + 12) == "^VOL SPLIT^\n")
        {
            sentenceStart = x + 12;
            volSplit.push(numSentences);
        }
        */

        //Last sentence
        if(x == fileContent.length - 1)
        {
            sentences.push(fileContent.slice(sentenceStart, x + 1));
            numSentences++;
        }

        let segments = Math.ceil(fileContent.length / 1000);
        if(x % segments == 0)
        {
            console.log("Process character: " + x + "/" + fileContent.length);
        }
    }

    //Fill out breaks
    for(let x = 0; x < numSentences; x++)
    {
        if((sentences[x].replace("\n", "")).replace("\r", "") == "^PART SPLIT^")
        {
            partSplit.push(x);
        }
        else if((sentences[x].replace("\n", "")).replace("\r", "") == "^CHAPTER SPLIT^")
        {
            chapterSplit.push(x);
        }
        else if((sentences[x].replace("\n", "")).replace("\r", "") == "^VOL SPLIT^")
        {
            volSplit.push(x);
        }
    }

    /*
    //Fill out breaks
    for(let x = 0; x < numSentences; x++)
    {
        console.log(nextSentence);
        let nextSentence = sentences[x];
        nextSentence = nextSentence.replace("\n", "");
        nextSentence = nextSentence.replace("\r", "");
        if(nextSentence == "^PART SPLIT^")
        {
            partSplit.push(x);
        }
        else if(nextSentence == "^CHAPTER SPLIT^")
        {
            chapterSplit.push(x);
        }
        else if(nextSentence == "^VOL SPLIT^")
        {
            volSplit.push(x);
        }
    }
    */

    let output = 
    {
        "volumes":
        [
            {
                "chapters": 
                [
                    {
                        "parts":
                        [
                            {
                                "sentences":
                                [
                                    /*
                                    {
                                        "text": "",
                                        "speaker": null,
                                        "isDialogue": false,
                                        "isNarration": false,
                                        "isTLNote": false
                                    }
                                    */
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    };

    let currentVol = 0;
    let currentChapter = 0;
    let currentPart = 0;
    let currentSentence = 0;
    for(let x = 0; x < numSentences; x++)
    {
        //Split a new volume
        if(volSplit.includes(x))
        {
            currentVol++;
            currentSentence = 0;
            currentPart = 0;
            currentChapter = 0;
            output.volumes[currentVol] = 
            {
                "chapters": 
                [
                    {
                        "parts": 
                        [
                            {
                                "sentences": []
                            }
                        ]
                    }
                ]
            };
            continue;
        }
        //Split a new chapter
        if(chapterSplit.includes(x))
        {
            currentChapter++;
            currentSentence = 0;
            currentPart = 0;
            output.volumes[currentVol].chapters[currentChapter] = 
            {
                "parts": 
                [
                    {
                        "sentences": []
                    }
                ]
            };
            continue;
        }
        //Split a new part
        if(partSplit.includes(x))
        {
            currentPart++;
            currentSentence = 0;
            output.volumes[currentVol].chapters[currentChapter].parts[currentPart] = 
            {
                "sentences": []
            };
            continue;
        }

        //https://stackoverflow.com/questions/9401312/how-to-replace-curly-quotation-marks-in-a-string-using-javascript
        let finalSentence = sentences[x]
            .replace("\n", "")
            .replace("\r", "")
            .replace(/[\u2018\u2019]/g, "'")
            .replace(/[\u201C\u201D]/g, '"');
            
        /*
        let narration = false;
        let dialogue = false;
        let tlnote = false;
        let quote = false;
        */
       let type = "none";
        if(finalSentence[0] == "\"")
        {
            //dialogue = true;
            type = "dialogue";
        }
        else if(finalSentence[0] == "<" && finalSentence[1] == "\"") //Exception for the character Aegis/Aigis
        {
            type = "dialogue";
        }
        else if(finalSentence[0] == "<")
        {
            //tlnote = true;
            type = "tlnote";
        }
        else if(finalSentence[0] == "'")
        {
            //quote = true;
            type = "quote";
        }
        else
        {
            type = "narration";
        }
        /*
        if(!dialogue && !tlnote && !quote)
        {
            //narration = true;
        }
        */
        

        output.volumes[currentVol].chapters[currentChapter].parts[currentPart].sentences[currentSentence] = 
        {
            //"text": (sentences[x].replace("\n", "")).replace("\r", ""),
            "text": finalSentence,
            //"speaker": null,
            /*
            "isDialogue": dialogue,
            "isNarration": narration,
            "isTLNote": tlnote,
            "isQuote": quote
            */
           "type": type
        };
        currentSentence++;

        //console.log("Wrapping sentence: " + x + "/" + numSentences + " | " + (sentences[x].replace("\n", "")).replace("\r", ""));

        let segments = Math.ceil(numSentences / 1000);
        if(x % segments == 0)
        {
            console.log("Wrapping sentence: " + x + "/" + numSentences);
        }
    };

    //Print the JSON results to HTML
    let final = JSON.stringify(output);
    if(printJSONResult)
    {
        //Print JSON result
        let indent = 0;
        for(let x = 0; x < final.length; x++)
        {
            document.getElementById("fileContents").innerHTML += final[x];
            if(final[x] == "{" || final[x] == "}" || final[x] == "[" || final[x] == "]" || final[x] == ",")
            {
                document.getElementById("fileContents").innerHTML += "<br />";
                if(final[x] == "{" || final[x] == "[")
                {
                    indent++;
                }
                else if(final[x] == "}" || final[x] == "]")
                {
                    indent--;
                }
                document.getElementById("fileContents").innerHTML += "----".repeat(indent);
            }
        }
    }

    /*
    //Print sentences to HTML Page
    for(let x = 0; x < numSentences; x++)
    {
        document.getElementById("fileContents").innerHTML += sentences[x] + "<br />";
    }
    //*/

    //https://stackoverflow.com/questions/19721439/download-json-object-as-a-file-from-browser
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(final);
    var dlAnchorElem = document.getElementById('downloadAnchorElem');
    dlAnchorElem.setAttribute("href",     dataStr     );
    dlAnchorElem.setAttribute("download", "data.json");
    dlAnchorElem.click();
}

function returnAnalysis(jsonFile)
{
    let text = JSON.parse(jsonFile);

    let output = "";

    let charVocab = {};
    let wordVocab = {};

    let analysisOutputToFile = document.getElementById("analysisOutput").checked;
    let lineDelimiter = "\n";
    if(!analysisOutputToFile)
    {
        lineDelimiter = "<br />";
    }

    let maskNarration = document.getElementById("narrationAnalysisMask").checked;
    let maskDialogue = document.getElementById("dialogueAnalysisMask").checked;
    let maskTLNote = document.getElementById("tlnoteAnalysisMask").checked;
    let maskQuote = document.getElementById("quoteAnalysisMask").checked;

    let outputCharDistToCSV = document.getElementById("csvChars").checked;
    let outputWordDistToCSV = document.getElementById("csvWords").checked;

    //General stats
    let numVols = 0;
    let numChapters = 0;
    let numParts = 0;
    let numSentences = 0;
    let numWords = 0;
    let numChars = 0;

    //Full traversal of data
    for(let vol = 0; vol < text.volumes.length; vol++)
    {
        numVols++;

        for(let chapter = 0; chapter < text.volumes[vol].chapters.length; chapter++)
        {
            numChapters++;

            for(let part = 0; part < text.volumes[vol].chapters[chapter].parts.length; part++)
            {
                numParts++;

                for(let sentence = 0; sentence < text.volumes[vol].chapters[chapter].parts[part].sentences.length; sentence++)
                {
                    numSentences++;

                    let type = text.volumes[vol].chapters[chapter].parts[part].sentences[sentence].type;

                    //Filter out specified sentences
                    if(maskNarration && type == "narration")
                    {
                        continue;
                    }
                    else if(maskDialogue && type == "dialogue")
                    {
                        continue;
                    }
                    else if(maskTLNote && type == "tlnote")
                    {
                        continue;
                    }
                    else if(maskQuote && type == "quote")
                    {
                        continue;
                    }
                    
                    let currentSentence = text.volumes[vol].chapters[chapter].parts[part].sentences[sentence].text;

                    for(let char = 0; char < currentSentence.length; char++)
                    {
                        numChars++;

                        //Character-wise Analysis
                        let currentChar = currentSentence[char];

                        //Count character occurence
                        if(charVocab[currentChar] === undefined)
                        {
                            charVocab[currentChar] = 1;
                        }
                        else
                        {
                            charVocab[currentChar]++;
                        }
                    }

                    //Sentence-wise Analysis

                    //Count word occurence
                    let cleanSentence = currentSentence.toLowerCase().replace(/[^A-Za-z0-9\s]/g, "").replace(/ +/g, " "); 
                    //Remove non-alphanumeric or whitespace, convert to lowercase, remove duplicate spaces
                    let words = cleanSentence.split(" "); //Split sentence by spaces

                    for(let word = 0; word < words.length; word++)
                    {
                        /*
                        if(word == "") //Strangely, I'm getting empty words in my occurences
                        {
                            console.log(cleanSentence);
                        }
                        */
                       numWords++;

                        if(wordVocab[words[word]] === undefined)
                        {
                            wordVocab[words[word]] = 1;
                        }
                        else
                        {
                            wordVocab[words[word]]++;
                        }
                    }

                    //console.log("Processing sentence: " + sentence + " in part: " + part + " in chapter: " + chapter + " in vol: " + vol);
                }

                console.log("Processing part: " + part + " in chapter: " + chapter + " in vol: " + vol);
            }
        }
    }

    let charVocabKeyValPairs = Object.keys(charVocab).map(function(key) {return [key, charVocab[key]];});
    let wordVocabKeyValPairs = Object.keys(wordVocab).map(function(key) {return [key, wordVocab[key]];});
    let charVocabKeyValPairsSorted = charVocabKeyValPairs.sort(function(a,b) {return b[1] - a[1];});
    let wordVocabKeyValPairsSorted = wordVocabKeyValPairs.sort(function(a,b) {return b[1] - a[1];});

    output += "Number of Volumes: " + lineDelimiter;
    output += numVols + lineDelimiter;
    output += "Number of Chapters: " + lineDelimiter;
    output += numChapters + lineDelimiter;
    output += "Number of Parts: " + lineDelimiter;
    output += numParts + lineDelimiter;
    output += "Number of Lines: " + lineDelimiter;
    output += numSentences + lineDelimiter;
    output += "Number of Words: " + lineDelimiter;
    output += numWords + lineDelimiter;
    output += "Number of Characters: " + lineDelimiter;
    output += numChars + lineDelimiter;
    output += lineDelimiter;
    output += "Average Characters per Line: " + lineDelimiter;
    output += (numChars / numSentences) + lineDelimiter;
    output += "Average Lines per Part: " + lineDelimiter;
    output += (numSentences / numParts) + lineDelimiter;
    output += "Average Characters per Word: " + lineDelimiter;
    output += (numChars / numWords) + lineDelimiter;
    output += "Average Words per Line: " + lineDelimiter;
    output += (numWords / numSentences) + lineDelimiter;
    output += lineDelimiter;
    output += "Number of Unique Characters: " + lineDelimiter;
    output += charVocabKeyValPairsSorted.length + lineDelimiter;
    output += "Number of Unique Words: " + lineDelimiter;
    output += wordVocabKeyValPairsSorted.length + lineDelimiter;

    //Non-US keyboard chars
    let nonStandChars = "";
    for(let x = 0; x < charVocabKeyValPairsSorted.length; x++)
    {
        nonStandChars += charVocabKeyValPairsSorted[x][0];
    }
    nonStandChars = nonStandChars.replace(/([\x20-\x7E])+/g, ""); //Replace all US keyboard chars
    output += "Number of Unique Non-Standard Characters (Non-US Keyboard): " + lineDelimiter;
    output +=  nonStandChars.length + lineDelimiter;
    output += "Non-Standard Characters: " + nonStandChars + lineDelimiter;
    let nonStandCharsNum = 0;
    for(let x = 0; x < charVocabKeyValPairsSorted.length; x++)
    {
        if(nonStandChars.includes(charVocabKeyValPairsSorted[x][0]))
        {
            nonStandCharsNum += charVocabKeyValPairsSorted[x][1];
        }
    }
    output += "Number of Non-Standard Characters: " + lineDelimiter;
    output +=  nonStandCharsNum + lineDelimiter;
    output += "Percentage of Non-Standard Characters: " + lineDelimiter;
    output += (nonStandCharsNum / numChars) + lineDelimiter;
    let wordCount = 0;
    for(let x = 0; x < wordVocabKeyValPairsSorted.length; x++)
    {
        if(x < 100)
        {
            wordCount += wordVocabKeyValPairsSorted[x][1];
        }
    }
    output += "Number of 100 Most Used Words: " + lineDelimiter;
    output += wordCount + lineDelimiter;
    output += "Percentage of 100 Most Used Words: " + lineDelimiter;
    output += (wordCount / numWords) + lineDelimiter;

    output += "-------------------------------------------------" + lineDelimiter;

    let csvOutput = "";

    output += "Character-wise Occurence:" + lineDelimiter;
    for(let x = 0; x < charVocabKeyValPairsSorted.length; x++)
    {
        output += charVocabKeyValPairsSorted[x][0] + " - " + charVocabKeyValPairsSorted[x][1] + lineDelimiter;
        if(outputCharDistToCSV)
        {
            if(charVocabKeyValPairsSorted[x][0] == "\"")
            {
                charVocabKeyValPairsSorted[x][0] = "\"\"";
            }
            csvOutput += "\"" + charVocabKeyValPairsSorted[x][0] + "\"," + charVocabKeyValPairsSorted[x][1] + "\n";
        }
    }
    
    output += "-------------------------------------------------" + lineDelimiter;
    output += "Word-wise Occurence:" + lineDelimiter;
    for(let x = 0; x < wordVocabKeyValPairsSorted.length; x++)
    {
        if(false)
        {
            console.log(JSON.stringify(wordVocabKeyValPairsSorted[x][0]));
        }
        output += wordVocabKeyValPairsSorted[x][0] + " - " + wordVocabKeyValPairsSorted[x][1] + lineDelimiter;
        if(outputWordDistToCSV)
        {
            csvOutput += "\"" + wordVocabKeyValPairsSorted[x][0] + "\"," + wordVocabKeyValPairsSorted[x][1] + "\n";
        }
    }

    if(outputCharDistToCSV || outputWordDistToCSV)
    {
        //https://stackoverflow.com/questions/19721439/download-json-object-as-a-file-from-browser
        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(csvOutput);
        var dlAnchorElem = document.getElementById('downloadAnchorElem');
        dlAnchorElem.setAttribute("href",     dataStr     );
        dlAnchorElem.setAttribute("download", "dist.csv");
        dlAnchorElem.click();
    }

    if(!analysisOutputToFile)
    {
        document.getElementById("fileContents").innerHTML = output;
    }
    else
    {
        //https://stackoverflow.com/questions/19721439/download-json-object-as-a-file-from-browser
        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(output);
        var dlAnchorElem = document.getElementById('downloadAnchorElem');
        dlAnchorElem.setAttribute("href",     dataStr     );
        dlAnchorElem.setAttribute("download", "stats.txt");
        dlAnchorElem.click();
    }
}