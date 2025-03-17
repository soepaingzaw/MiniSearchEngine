# Mini Seach Engine Project

#### Project Description

Created a thesaurus that is specific to legal-context by scraping dictionary.law.com,
forming a mapping from a legal term to a set of related terms. Performed stemming
on both the key and value since already performing stemming on the query (otherwise the 
query and thesaurus would not match).

## Project Design

### Indexing

Main entry point is `index.py`. Helper files are `InputOutput.py`, `Tokenizer.py`.

Indexing approach => positional indexing.

## Project style and setup

### Project setup

Use whatever CPython interpreter as long as you are on 3.8.10 and have NLTK (with `punkt` downloaded).
