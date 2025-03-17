from __future__ import annotations

import ctypes as ct

import csv
import nltk
import string
import Config

from Types import *
from typing import List, Set

STEMMER = nltk.stem.porter.PorterStemmer()
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))  # found a fix from StackOverflow for field size too small


def make_doc_read_generator(in_file: str, stop_words_file: str) -> TermInfoTupleGenerator:
    """
    Generator function for the next (term, term_pos, doc_length, doc_id) tuple.
    Call this function to make the generator first, then use next() to generate the next tuple.
    Skips over stop words.
    Yields (None, None, None, None) when done.
    :param in_file: The name of the input file
    :param stop_words_file: The name of the stop words file.
    :return: A generator object for the term information tuple (see above)
    """

    # read the provided stop words file ONCE and keep stop words in memory
    with open(stop_words_file, 'r') as f:
        stopwords = set(f.read().split())

    already_read: Set[DocId] = set()
    with open(in_file, mode='r', encoding='utf-8', newline='') as doc:
        doc_reader = csv.reader(doc)
        for i, row in enumerate(doc_reader):
            if i == 0:
                continue  # we skip the first row (headers)
            elif i % 50 == 0:
                print("progress:", i)

            doc_id, title, content, date_posted, court = row

            # since duplicates exist in the corpus, here we skip doc IDs already processed
            if int(doc_id) in already_read:
                continue
            else:
                already_read.add(int(doc_id))

            title_tokens = tokenize(title, "title", stopwords)
            date_tokens = tokenize(date_posted, "date", stopwords)
            court_tokens = tokenize(court, "court", stopwords)

            # when zone is "content", perform extra parsing
            # for this, the name of the court is required so pass it in as a param
            content_tokens = tokenize(content, "content", stopwords, court=court)

            tokens = title_tokens + content_tokens + date_tokens + court_tokens
            doc_length = len(tokens)

            # for this assignment, we can assume that document names are integers without exception
            # since we are using a generator, we only count the number of tokens once per file
            for term_pos, term in enumerate(tokens):
                yield term, term_pos, doc_length, int(doc_id)
    yield None, None, None, None


def tokenize(doc_text: str,
             zone: str,
             stop_words: Set[str],
             court: Optional[str] = None) -> List[str]:
    """
    Takes in document text and tokenizes.
    Also does post-tokenization cleaning like stemming.
    :param doc_text: The text to be tokenized
    :param zone: The zone the text is associated with
    :param stop_words: The set of stop words to be used
    :param court: The name of the court (only included when zone is "content")
    :return: List of tokens
    """

    # case folding
    doc_text = doc_text.lower()

    # tokenize and stem
    tokens = nltk.tokenize.word_tokenize(doc_text)
    tokens = [STEMMER.stem(tok) for tok in tokens]

    # remove tokens that are purely punctuation
    def is_not_only_punct(tok): return any(char not in string.punctuation for char in tok)
    tokens = [tok for tok in tokens if is_not_only_punct(tok)]

    # remove stopwords from the tokens and add delimiter for zones
    tokens = [word for word in tokens if word not in stop_words]

    # each token will have the zone appended to the front using the @ symbol
    # for example, "title@token", "content@token"
    if zone == "content":
        tokens = create_zones(tokens, court)
    else:
        tokens = [zone + '@' + token for token in tokens]

    return tokens


def create_zones(tokens: List[Term], court: str) -> List[Term]:
    """
    Given a list of raw terms, parse them with the specific parser for its court and
    tag the content with more accurate tags.
    :param tokens: The list of input terms
    :param court: The name of the court as a string
    :return: The list of tagged terms
    """
    
    # handle the case where we have not created a special parsing config for the court
    # this is only the case for the less frequently occurring courts
    if court not in Config.PARSING_CONFIG:
        return ["content@" + tok for tok in tokens]
    court_field = Config.PARSING_CONFIG[court]
    
    # find number of words for that specific court
    section_num_words = court_field['num_words']
    section_keywords = [STEMMER.stem(tok) for tok in court_field['section'].split(', ')]
    parties_num_words = court_field['parties_num_words']
    parties_keywords = [STEMMER.stem(tok) for tok in court_field['parties'].split(', ')]
    
    term_list = []
    
    # go through each token and tag it!
    # upon encountering a token that's in either keyword set, start a run of
    # "section@" or "parties@" tagging for the specified duration
    remaining_section = 0
    remaining_parties = 0
    for tok in tokens:
        if remaining_section == 0 and remaining_parties == 0:
            if tok in section_keywords:
                remaining_section = section_num_words
            elif tok in parties_keywords:
                remaining_parties = parties_num_words
        if remaining_section:
            remaining_section -= 1
            term_list += ["section@" + tok]
        elif remaining_parties:
            remaining_section -= 1
            term_list += ["parties@" + tok]
        else:
            term_list += ["content@" + tok]
            
    return term_list


def clean_query_token(token: str) -> str:
    """
    Case-folds and stems a single token.
    :param token: The token to be case-folded and stemmed
    :return: Case-folded and stemmed token
    """
    return STEMMER.stem(token.lower())


def tokenize_query(query: str) -> List[str]:
    """
    Takes in a string and performs tokenization, stemming and case-folding.
    If the string contains a phrasal query (terms enclosed by ""), then leave it as a single
    token with the terms within it stemmed. If the token is AND/OR, then leave it as is.
    :param query: The string to process
    :return: A list of processed tokens
    """
    tokenizer = nltk.RegexpTokenizer(r'\w+|\s+|"[^"]+"')
    tokens = tokenizer.tokenize(query)
    result = []
    for token in tokens:
        token = token.strip().strip('"')
        if not token:
            continue
        elif token == "AND":
            result.append(token)
        elif ' ' in token:  # if the token is a phrase,
            stemmed_phrase = ' '.join([STEMMER.stem(subtoken.strip().casefold()) for subtoken in token.split()])
            result.append(stemmed_phrase)
        else:  # if the token is a single word
            result.append(STEMMER.stem(token.strip().casefold()))
    return result
