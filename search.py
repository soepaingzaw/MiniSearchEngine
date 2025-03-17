#!/usr/bin/python3
import pickle
import argparse

from typing import List
from Tokenizer import tokenize_query
from QueryRefinement import expand_query, tag_query_with_zones, extract_date
from Searcher import search_freetext_query, search_boolean_query
from Types import *
import Config


def run_search(dict_file: str, postings_file: str, queries_file: str, results_file: str):
    """
    using the given dictionary file, postings file, and optionally 
    thesaurus pickle file, perform searching on the given queries 
    file and output the results to the results file
    """
    print("running search on the queries...")

    # READ UTILITY FILES
    pointer_dct: Dict[Term, int]
    docs_len: Dict[DocId, DocLength]
    champion_dct: Dict[DocId, List[Tuple[Term, TermWeight]]]

    with open(dict_file, 'rb') as df,\
         open(Config.LENGTHS_FILE, 'rb') as lf,\
         open(Config.CHAMPION_FILE, "rb") as cf:
        pointer_dct = pickle.load(df)
        docs_len = pickle.load(lf)
        champion_dct = pickle.load(cf)

    # READ QUERY FILE
    query_tokens: List[str]
    relevant_docs: List[DocId] = []
    with open(queries_file, "r") as qf:
        query = qf.readline()
        while relevant_doc := qf.readline().strip():
            relevant_docs.append(int(relevant_doc))

    # QUERY PROCESSING
    # extract a single date from the query, if it exists
    # otherwise, extracted_dates will be an empty list
    extracted_dates: List[str] = extract_date(query)

    # tokenize the query
    query_tokens: List[str] = tokenize_query(query)

    # handle case where it is a phrasal query and boolean query
    is_boolean_query = 'AND' in query_tokens

    search_output: List[DocId] = []

    print(f"running {'non-' if not is_boolean_query else ''}boolean query")

    # HANDLE BOOLEAN QUERY
    if is_boolean_query:
        search_output = search_boolean_query(query_tokens, pointer_dct, postings_file)

    # HANDLE FREE TEXT QUERY
    # There is the possibility of running this as a backup in case the boolean query returns nothing
    if (not is_boolean_query) or (not search_output):
        if is_boolean_query:
            print("boolean query failed! defaulting to free text")
        # CONVERT PHRASAL QUERIES INTO FREE TEXT
        # For now, we're not sure how to handle phrasal queries in free text...
        all_tokens = []
        for tok in query_tokens:
            if ' ' in tok:
                all_tokens += tok.split()
            else:
                all_tokens += [tok]

        # QUERY EXPANSION
        if Config.RUN_QUERY_EXPANSION:
            with open(Config.THESAURUS_FILENAME, "rb") as tf:
                thesaurus = pickle.load(tf)
            all_tokens = expand_query(all_tokens, thesaurus)

        # TAGGING QUERY WITH ZONES
        all_tokens = sum(tag_query_with_zones(all_tokens), [])  # flatten list
        all_tokens += ["date@" + date for date in extracted_dates]  # add in any dates extracted

        # SEARCHING
        search_output: List[DocId]
        search_output = search_freetext_query(all_tokens,
                                              pointer_dct,
                                              docs_len,
                                              postings_file,
                                              relevant_docs,
                                              champion_dct)

    # DEBUG PRINTS
    # true_pos = sum([rd in search_output for rd in relevant_docs])
    # precision = true_pos / len(search_output) if len(search_output) > 0 else 0
    # recall = true_pos / len(relevant_docs)
    # f2_score = 5 * (precision * recall) / (4*precision + recall)
    # def find_item(lst, item):
    #     try:
    #         return lst.index(item)
    #     except:
    #         return -2
    # print("Positions of results:", [1+find_item(search_output, rd) for rd in relevant_docs])
    # print(f"Precision: {precision}, Recall: {recall}, F2: {f2_score}")

    output = " ".join(map(str, search_output))
    print("docs found:", len(search_output))
    with open(results_file, "w") as rf:
        rf.write(output)


# python3 search.py -d dictionary.txt -p postings.txt -q queries/q1.txt -o results.txt
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', dest='dictionary_file', required=True)
    arg_parser.add_argument('-p', dest='postings_file', required=True)
    arg_parser.add_argument('-q', dest='queries_file', required=True)
    arg_parser.add_argument('-o', dest='output_file', required=True)
    args = arg_parser.parse_args()

    run_search(args.dictionary_file, args.postings_file, args.queries_file, args.output_file)
