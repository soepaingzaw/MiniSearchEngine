# Mini Search Engine Project

#### Project Description

Created a thesaurus that is specific to legal-context by scraping dictionary.law.com,
forming a mapping from a legal term to a set of related terms. Performed stemming
on both the key and value since already performing stemming on the query (otherwise the 
query and thesaurus would not match).

## Project Design


- **[Config.py](c:\Users\User\Documents\Native\MiniSeachEngine\miniSearchEngine\Config.py)**: Contains configuration parameters for the search engine, such as file paths, query expansion settings, and relevance feedback coefficients.

- **[index.py](c:\Users\User\Documents\Native\MiniSeachEngine\miniSearchEngine\index.py)**: Main entry point for indexing documents. It builds the index from documents, calculates document lengths, and creates a champion list of significant terms for each document.

- **[InputOutput.py](c:\Users\User\Documents\Native\MiniSeachEngine\miniSearchEngine\InputOutput.py)**: Handles reading and writing of posting lists and other data structures. Includes the `PostingReader` class for reading posting lists and functions for writing the index to disk.

- **[QueryRefinement.py](c:\Users\User\Documents\Native\MiniSeachEngine\miniSearchEngine\QueryRefinement.py)**: Implements query refinement techniques such as Rocchio algorithm for relevance feedback and query expansion using a thesaurus.

- **[Searcher.py](c:\Users\User\Documents\Native\MiniSeachEngine\miniSearchEngine\Searcher.py)**: Contains functions for performing boolean and free-text searches. It calculates query and document vectors and ranks documents based on their relevance to the query.

- **[search.py](c:\Users\User\Documents\Native\MiniSeachEngine\miniSearchEngine\search.py)**: Main entry point for running searches. It reads the dictionary and postings files, processes queries, and outputs search results.

- **[stopwords.txt](c:\Users\User\Documents\Native\MiniSeachEngine\miniSearchEngine\stopwords.txt)**: A text file containing a list of stop words to be excluded from indexing and searching.

- **[Tokenizer.py](c:\Users\User\Documents\Native\MiniSeachEngine\miniSearchEngine\Tokenizer.py)**: Contains functions for tokenizing and stemming text, as well as generating term information tuples for indexing.

- **[Types.py](c:\Users\User\Documents\Native\MiniSeachEngine\miniSearchEngine\Types.py)**: Defines type aliases and data structures used throughout the project, such as `DocId`, `Term`, and `Vector`.

### Indexing

Main entry point is `index.py`. Helper files are `InputOutput.py`, `Tokenizer.py`.

Indexing approach => positional indexing.

## Project style and setup

### Project setup

Use whatever CPython interpreter as long as you are on 3.8.10 and have NLTK (with `punkt` downloaded).
