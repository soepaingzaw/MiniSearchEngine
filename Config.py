# Configuration file for various parameters

# CHAMPION LIST FILE
K: int = 1000
CHAMPION_FILE: str = "champion.txt"

# LENGTHS FILE
LENGTHS_FILE: str = "lengths.txt"

# STOP WORDS FILE
STOP_WORDS_FILE: str = "stopwords.txt"

# POSITIONAL INDICES
WRITE_POS: bool = True

# QUERY EXPANSION
RUN_QUERY_EXPANSION: bool = True
THESAURUS_FILENAME: str = "stemmed_thesaurus.pickle"

# RELEVANCE FEEDBACK
# Classically, beta is smaller than alpha since the original query should carry a higher weightage.
# But for this dataset, we perform a lot of query expansion, so our query itself is not likely to be that accurate.
# We rely heavily on the relevant documents provided to narrow down our results, so make beta higher than alpha.
RUN_ROCCHIO: bool = True
ALPHA: float = 0.8  # determined through experimentation
BETA: float = 1.1

# QUERY PRUNING
# Remove any terms with weight below the threshold (max_weight/threshold_val) from the query vector
# We are pruning AFTER relevance feedback, so relevant terms should have gained more weight and vice versa for
# non-relevant terms. Thus, we can safely put the threshold somewhere around 4 (= 1/4 of max weight).
RUN_QUERY_PRUNING: bool = False  # DISABLED FOR NOW
PRUNING_THRESHOLD: float = 4

# CONTENT PARSING
PARSING_CONFIG = {
    'NSW Court of Criminal Appeal': {
        'section': 'section, act',
        'num_words': 15,  # estimate number of words in a section
        'parties': 'parties, witness, victims, appellent, ms, mr',
        'parties_num_words': 3
    },
    'NSW Supreme Court': {
        'section': 'penalty, act, force',
        'num_words': 15,
        'parties': 'parties, witness, victims, appellent, ms, mr, offender, judgememt',
        'parties_num_words': 7
    },
    'CA Supreme Court': {
        'section': 'section, act, sect',
        'num_words': 10,
        'parties': 'respondent, present, defendant, prosecutrix, prosecutor, sir, plaintiff, named, acquainted',
        'parties_num_words': 3
    },
    'NSW District Court': {
        'section': 'section, act',
        'num_words': 15,
        'parties': 'parties, sir, mr, ms, witness, victims, appellent',
        'parties_num_words': 3
    },
    'SG High Court': {
        'section': 'section, act, case, number, crime, s, ss, CPC, penal, code',
        'num_words': 10,
        'parties': 'parties, judge, counsel, name(s), coram',
        'parties_num_words': 5
    },
    'High Court of Australia': {
        'section': 'act, code, statutes',
        'num_words': 10,
        'parties': 'parties, lawyer, high, court, australia, appellant, respondent, representation, intervener',
        'parties_num_words': 10
    },
    'Federal Court of Australia': {
        'section': 'legislation, section, act, catchwords',
        'num_words': 10,
        'parties': 'parties, judges, respondent, mr, ms',
        'parties_num_words': 5
    },
    'SG Court of Appeal': {
        'section': 'section, act, case, number, crime, s, ss, CPC, penal, code',
        'num_words': 10,
        'parties': 'parties, judge, counsel,coram',
        'parties_num_words': 5
    },
    'NSW Court of Appeal': {
        'section': 'section, act, charges, order',
        'num_words': 15,
        'parties': 'judgement, mr, ms, applicant, counsel, citation, complainant',
        'parties_num_words': 5
    },
    'UK Crown Court': {
        'section': 'convict, act, charges, honourable',
        'num_words': 10,
        'parties': 'mr, ms, applicant, counsel, victim, complainant, ‐v‐',
        'parties_num_words': 5
    },
    'SG District Court': {
        'section': 'section, act, case, number, crime, s, ss, CPC, penal, code',
        'num_words': 10,
        'parties': 'parties, judge, counsel, name(s), coram, prosecutor, you, ms, mr',
        'parties_num_words': 5
    },
    'UK Court of Appeal': {
        'section': 'apprehended, act, section',
        'num_words': 10,  # estimate number of words in a section
        'parties': 'between, witness, before, honour, appellent, ms, mr, ‐v‐, respondent',
        'parties_num_words': 4
    },
    'NSW Land and Environment Court': {
        'section': 'offence, section, repealed',
        'num_words': 10,  # estimate number of words in a section
        'parties': 'witness, council, appellent, ms, mr, defendant, respondent',
        'parties_num_words': 5
    },
    'UK High Court': {
        'section': 'offence, section, repealed',
        'num_words': 10,  # estimate number of words in a section
        'parties': 'witness, council, claimant, ms, mr, -v-, defendant',
        'parties_num_words': 10
    },
    'SG Privy Council': {
        'section': 'section, act, case, number, crime, s, ss, CPC, penal, code',
        'num_words': 10,  # estimate number of words in a section
        'parties': 'witness, sir, lord, ms, mr, parties, defendant, respondent, prosecutor',
        'parties_num_words': 10
    },
    'Singapore International Commercial Court': {
        'section': 'section, act, case, number, crime, s, ss, CPC, penal, code',
        'num_words': 10,
        'parties': 'parties, judge, counsel, coram, lord',
        'parties_num_words': 5
    }
}
