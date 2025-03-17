from __future__ import annotations

from InputOutput import PostingReader
from QueryRefinement import run_rocchio
from math import log10
from typing import List, Set
from Types import *

import Config


def search_boolean_query(query_tokens: List[str],
                         pointer_dct: Dict[Term, int],
                         postings_file: str) -> List[DocId]:
    """
    Perform a boolean query based on the given query tokens.
    Will try to find a perfect match, but failing that, will return the last best result
    Return a list of relevant document IDs.
    :param query_tokens: List of query tokens
    :param pointer_dct: Dictionary of term -> postings list pointer
    :param postings_file: Name of the postings list file
    :return: List of relevant document IDs
    """

    subqueries = [subquery for subquery in query_tokens if not subquery == 'AND']

    all_search_outputs: List[DocId] = []
    for subquery in subqueries:
        # PERFORM PHRASAL QUERY SEARCH FOR CURRENT SUBQUERY
        if " " in subquery:  # MULTI-WORD PHRASAL QUERY
            intermediate_search_outputs = search_phrasal_query(subquery, pointer_dct, postings_file)
        else:  # SINGLE WORD PHRASAL QUERY
            query_word = subquery
            posting_list = {}
            for zone in ("content@", "title@", "court@", "parties@", "section@"):
                temp_word = zone + query_word
                if temp_word in pointer_dct:
                    posting_list.update(get_posting_list(postings_file, pointer_dct, temp_word))

            if posting_list:  # if term exists in corpus,
                intermediate_search_outputs = [doc_id for doc_id in posting_list.keys()]
            else:  # if term does not exist in corpus,
                intermediate_search_outputs = []

        # EARLY TERMINATION
        if not intermediate_search_outputs:
            # if current subquery gives no results, terminate early
            break

        # INTERSECT WITH EXISTING RESULTS
        if all_search_outputs:
            intersected = boolean_and(intermediate_search_outputs, all_search_outputs)
            if not intersected:
                # early termination condition #2 -- if intersect yields nothing
                break
        else:
            all_search_outputs += intermediate_search_outputs

    return all_search_outputs


def search_phrasal_query(phrasal_query: str,
                         pointer_dct: Dict[Term, int],
                         postings_file: str) -> List[DocId]:
    """
    Using the PostingReader interface, process a given phrasal query by getting each term's
    posting list and intersecting the positional indices.
    :param phrasal_query: The phrase query to search
    :param pointer_dct: A dictionary of terms -> postings list pointer
    :param postings_file: The name of the postings list file
    :return: A list of relevant document IDs
    """

    # NOTE! Phrasal queries will come as a single string, so we need to split it
    # Also, phrasal query terms will contain NO ZONES! They will be temporarily added later
    phrasal_query = phrasal_query.split()

    with PostingReader(postings_file, pointer_dct) as pf:
        term_data = {}
        for query_term in phrasal_query:
            # we temporarily add zone tags just to extract the relevant posting lists
            # for example, posting list for "content@test", "title@test", ... will be
            # condensed into a single posting list and saved in term_data
            for zone in ("content@", "title@", "court@", "parties@", "section@"):
                temp_term = zone + query_term
                if temp_term not in pointer_dct:
                    continue
                pf.seek_term(temp_term)
                while not pf.is_done():
                    doc_id, _, term_pos = pf.read_entry()
                    doc_id_to_pos = term_data.get(query_term, {})
                    term_pos_set = doc_id_to_pos.get(doc_id, set())
                    term_pos_set.add(term_pos)
                    doc_id_to_pos[doc_id] = term_pos_set
                    term_data[query_term] = doc_id_to_pos

    # INTERSECTION OF POSITIONAL INDICES
    result: Dict[DocId, Set[TermPos]] = dict()
    for i, query_term in enumerate(phrasal_query):
        if query_term not in term_data:  # if either term not in corpus or term is a stop word, skip
            continue
        if not result:  # if initializing result
            result.update(term_data[query_term])
        else:
            # find documents of the current query term which are present in the result
            intersecting_docs = result.keys() & term_data[query_term].keys()
            # find documents which have gap of i
            new_result = {}
            for doc_id in intersecting_docs:
                term_pos_of_doc_set = term_data[query_term][doc_id]
                # subtract all term pos by i so that can we use set intersection
                shifted_term_pos_set = {term_pos - i for term_pos in term_pos_of_doc_set}
                # intersect our existing document term pos set with the current term's shifted term pos set
                # then, store the results
                new_term_pos_set = result[doc_id] & shifted_term_pos_set
                if new_term_pos_set:  # if the intersection with the new doc is not empty, write it to new_results
                    new_result[doc_id] = new_term_pos_set
            if not new_result:
                # early termination
                result = dict()
                break
            result = new_result
    # We only want the document IDs to be returned
    result_docs: List[DocId] = list(result.keys())
    return result_docs


def search_freetext_query(query_tokens: List[Term],
                          dictionary: Dict[Term, int],
                          docs_len_dct: Dict[DocId, DocLength],
                          postings_file: str,
                          relevant_docs: List[DocId],
                          champion_dct: Dict[DocId, List[Tuple[Term, TermWeight]]]
                          ) -> List[DocId]:
    """
    Using the PostingReader interface, process a given free text query by calculating scores
    for each document from its posting list, then returning the documents with the highest scores.
    :param query_tokens: The free text query as a list of tokens
    :param dictionary: A dictionary of terms to their positions in the postings list
    :param docs_len_dct: A dictionary of doc ID to doc length
    :param postings_file: The name of the postings list file
    :param relevant_docs: The list of relevant document IDs (from the query file)
    :param champion_dct: The champion list as a dictionary
    :return: A list of relevant document IDs
    """

    all_query_terms = set(query_tokens)
    dictionary_terms = set(dictionary.keys())
    query_terms = list(all_query_terms & dictionary_terms)
    N = len(docs_len_dct)

    # CALCULATE QUERY VECTOR
    query_vector: Vector
    query_vector = calc_query_vector(postings_file, dictionary, query_terms, N)

    # REFINE QUERY VECTOR W/ ROCCHIO ALGO
    if Config.RUN_ROCCHIO:
        query_vector = run_rocchio(
            Config.ALPHA, Config.BETA, champion_dct, relevant_docs, query_vector)

    # QUERY PRUNING
    # We remove all query terms below a certain threshold, set as a fraction of the top weight
    # This is only helpful when we run query expansion due to the large number of query tokens, so we include that
    # as a condition here to prevent accidental triggering
    if Config.RUN_QUERY_PRUNING and Config.RUN_QUERY_EXPANSION:
        # print(f"avg weight in query: {sum(query_vector.values())/len(query_vector)}")
        # print(f"total num of weights: {len(query_vector)}")
        threshold = max(query_vector.values()) / Config.PRUNING_THRESHOLD
        # print(f"num terms above threshold: {sum([w > threshold for w in query_vector.values()])}")
        query_vector = {term: weight for term, weight in query_vector.items() if weight >= threshold}
        query_terms = set(query_vector.keys())

    # CALCULATE DOCUMENT VECTOR
    doc_vector_dct: Dict[DocId, Vector]
    doc_vector_dct = calc_doc_vectors(postings_file, dictionary, query_terms)

    # ADDING WEIGHTS TO ZONES
    def apply_weights_to_vector(vct: Vector) -> Vector:
        for term in vct:
            if term.startswith("date"):  # dates and titles should be boosted since they are important
                vct[term] *= 1.5
            elif term.startswith("title"):
                vct[term] *= 1.5
            elif term.startswith("content"):
                pass  # multiplier is 1.0
            else:  # other sections can be ranked lower than the content itself
                vct[term] *= 0.8
        return vct

    for docID, vector in doc_vector_dct.items():
        doc_vector_dct[docID] = apply_weights_to_vector(vector)
    query_vector = apply_weights_to_vector(query_vector)

    # calculate the final scores for each document
    doc_scores: List[Tuple[DocId, float]] = []
    for doc_id in doc_vector_dct.keys():
        dot_prod = 0.
        for t in query_terms:
            d_score = doc_vector_dct[doc_id].get(t, 0.)
            dot_prod += d_score * query_vector[t]
        score = dot_prod / docs_len_dct[doc_id]  # normalization
        doc_scores.append((doc_id, score))

    # sort by ascending document ID first, then by descending score
    # since Python's sort is stable, we end up with a list sorted by
    # descending scores and tie-broken by ascending document IDs
    # return the top 10 in the list
    doc_scores = sorted(doc_scores, key=lambda x: x[0])
    doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)

    return [x[0] for x in doc_scores]  # we only want to keep the doc IDs!


def get_posting_list(postings_file, pointer_dct, query_term) -> Dict[DocId, Set[TermPos]]:
    """
    Gets the posting list of a single term from the postings list file as a dictionary.
    :param postings_file: The name of the posting file
    :param pointer_dct: The dictionary of term -> postings file pointer
    :param query_term: The term to retrieve the posting list of
    :return: Posting list as a dictionary of doc ID to a set of term positions
    """
    with PostingReader(postings_file, pointer_dct) as pf:
        pf.seek_term(query_term)
        doc_id_to_pos = {}
        while not pf.is_done():
            doc_id, _, term_pos = pf.read_entry()
            pos_set = doc_id_to_pos.get(doc_id, set())
            pos_set.add(term_pos)
            doc_id_to_pos[doc_id] = pos_set
    return doc_id_to_pos


def boolean_and(lst_1: List[DocId], lst_2: List[DocId]):
    """
    And operation to find intersect between two lists WITHOUT skip pointers
    :param lst_1: list of IDs in first list
    :param lst_2: list of IDs in second list
    :return: list of IDs in both lists
    """
    return sorted(list(set(lst_1).intersection(set(lst_2))))


# TODO: Do we need this function?
# def get_sub_list(list_of_tuple: List[Tuple[DocId, int]] | List[DocId], index: int) -> List[DocId]:
#     """
#     Extract list from list of tuple
#
#     :param list_of_tuple: List of tuple, e.g. [(1,3), (2,None), (3,5), (4,None), (5, None)]
#     :param index: Index at which you want to get the element from
#     :return: list e.g [1, 2, 3, 4, 5]
#     """
#
#     if not len(list_of_tuple):
#         return []
#
#     if type(list_of_tuple[0]) is not tuple:
#         return list_of_tuple
#
#     return list(zip(*list_of_tuple))[index]


def calc_query_vector(postings_file: str,
                      pointer_dct: Dict[Term, int],
                      query_terms: List[str],
                      n: int) -> Vector:
    """
    Calculates a query vector based on given query terms.
    Query vector will be in the form of a dictionary of term -> weight.
    :param postings_file: The name of the postings file
    :param pointer_dct: The dictionary of term -> postings file pointer
    :param query_terms: The list of query terms
    :param n: The total number of documents
    :return: The query vector
    """
    query_vector: Vector = dict()

    with PostingReader(postings_file, pointer_dct) as pf:
        for term in query_terms:
            query_vector[term] = calc_query_tfidf(
                term, query_terms, n, pf)

    return query_vector


def calc_doc_vectors(postings_file: str,
                     pointer_dct: Dict[Term, int],
                     query_terms: List[str]) -> Dict[DocId, Vector]:
    """
    Calculates a doc vectors based on given query terms.
    Document vectors will be in the form of a dictionary of (doc ID, term) -> weight.
    :param postings_file: The name of the postings file
    :param pointer_dct: The dictionary of term -> postings file pointer
    :param query_terms: The set of query terms
    :return: The document vector dictionary
    """
    doc_score_dct: Dict[Term, Dict[DocId, TermWeight]] = dict()
    doc_vector_dct: Dict[DocId, Vector]

    with PostingReader(postings_file, pointer_dct) as pf:
        for query_term in query_terms:
            doc_score_dct[query_term] = get_doc_tfidf_dict(query_term, pf)

    doc_vector_dct = invert_nested_dict(doc_score_dct)
    return doc_vector_dct


def invert_nested_dict(original: Dict) -> Dict:
    """
    Simple utility function.
    If a dictionary is originally dct[a][b] = c, this function inverts it to become dct[b][a] = c instead.
    :param original: Original dictionary, should be nested by one layer
    :return: Inverted dictionary as described
    """
    flipped = dict()
    for key, nested_dct in original.items():
        for subkey, val in nested_dct.items():
            if subkey not in flipped:
                flipped[subkey] = {key: val}
            else:
                flipped[subkey][key] = val
    return flipped


def calc_query_tfidf(term: Term,
                     query_terms: List[Term],
                     n: int,
                     pf: PostingReader) -> float:
    """
    Calculates the tf-idf weight of a term in a query.
    Takes in the posting file reader, N (total number of docs), and the list of tokens in the query.
    Returns a SINGLE tf-idf weight for that term and query.
    :param term: The term itself
    :param query_terms: List of tokens representing the query
    :param n: Total number of documents
    :param pf: The postings file reader interface
    :return: The calculated weight
    """
    pf.seek_term(term)
    df = pf.get_doc_freq()  # document frequency of term
    idf = log10(n / df)  # inverse document freq of term

    term_freq = query_terms.count(term)
    weight = (1 + log10(term_freq)) * idf

    return weight


def get_doc_tfidf_dict(term: str,
                       pf: PostingReader) -> Dict[DocId, float]:
    """
    Calculates the tf-idf weight of a term, for all docs listed in its
    posting list.
    Takes in the posting file reader and N (total number of docs).
    Returns a dictionary of doc ID -> tf-idf weight for that term/doc.
    :param term: The term itself
    :param pf: The postings file reader interface
    :return: The dictionary of calculated weights for each doc ID
    """
    pf.seek_term(term)

    weight_dct: Dict[DocId, float] = dict()
    prev_doc = 0
    while pf.get_num_docs_remaining() > 0:
        doc_id, term_freq, _ = pf.read_entry()
        if doc_id != prev_doc:
            # since we read directly from posting list,
            # term freq will never be 0, so we can ignore that case
            weight_dct[doc_id] = 1 + log10(term_freq)
            prev_doc = doc_id

    return weight_dct
