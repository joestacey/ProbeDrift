"""
Dataset loading configurations for all supported datasets.

Each entry contains the dataset metadata fields used by the benchmark. Fields:
  dataset_path     - HuggingFace path (str or list)
  text_column      - input column name
  label_column     - target column name
  prompt           - prompt template used for eval and training
  description      - description template used for eval and training
  train_split      - HF split name for training data
  eval_split       - HF split name for evaluation data
  n_shot           - number of few-shot examples (default 0)
  few_shot_split   - split to draw few-shot examples from
  few_shot_prompt  - separate few-shot prompt template (instruct models)

"""

# ---------------------------------------------------------------------------
# Fixed few-shot indices (see dataset.py's `fewshot_indices` param): removes
# all RNG dependence from few-shot selection (previously order-dependent on
# whatever dataset's subsample() call happened immediately before it in a
# multi-dataset training chain).
# ---------------------------------------------------------------------------

# Hand-picked, pre-verified-safe indices into the
# keivalya/MedQuad-MedicalQnADataset 'train' split: no truncation, no '%'/
# HPO-list formatting, no "symptom"/"signs and" keywords, no embedded
# newlines or braces.
MED_QUAD_FEWSHOT_INDICES = [9489, 1284, 7244, 10626, 3836]

# Fixed indices into the trivia_qa 'train' split, sanity-checked as clean,
# ordinary trivia Q&A pairs with no truncation/formatting issues.
TRIVIA_QA_FEWSHOT_INDICES = [23668, 91162, 92963, 122756, 131647]

DATASET_CONFIG = {

    # -------------------------------------------------------------------------
    # Non-instruct eval datasets
    # -------------------------------------------------------------------------

    'sciq': {
        'dataset_path': 'sciq',
        'text_column': 'question',
        'label_column': 'correct_answer',
        'prompt': '\n\nQuestion: {question}\nAnswer: ',
        'description': (
            'The following are contexts and questions about them. Each context is '
            'followed by a question and its answer.\n\nContext: {context}'
        ),
        'train_split': 'train',
        'eval_split': 'validation',
    },

    'trivia_qa': {
        'dataset_path': ['trivia_qa', 'rc.nocontext'],
        'text_column': 'question',
        'label_column': 'answer',
        'prompt': 'Question: {question}\nAnswer: {answer}',
        'train_split': 'train',
        'eval_split': 'validation',
        'n_shot': 5,
        'train_n_shot': 5,
        'few_shot_split': 'train',
        'fewshot_indices': TRIVIA_QA_FEWSHOT_INDICES,
    },

    'qa': {
        'dataset_path': 'coqa',
        'text_column': 'questions',
        'label_column': 'answers',
        'prompt': '\n\nQuestion: {question}\nAnswer: {answer}',
        'description': (
            'The following are stories and questions about them. Each story is '
            'followed by a question and its answer.\n\nStory: {story}'
        ),
        'train_split': 'train',
        'eval_split': 'validation',
    },

    'pubmed_qa': {
        'dataset_path': 'bigbio/pubmed_qa',
        'text_column': 'QUESTION',
        'label_column': 'LONG_ANSWER',
        'prompt': (
            'The following are abstracts and questions about them. Each abstract is '
            'followed by a question and its answer. '
            'Abstract: {context}\nQuestion: {question}\nAnswer: '
        ),
        'train_split': 'train',
        'eval_split': 'validation',
    },

    'xsum': {
        'dataset_path': 'xsum',
        'text_column': 'document',
        'label_column': 'summary',
        'prompt': "Here's the text and its short summary.\n\nText:\n{text}\n\nSummary (one sentence):\n",
        'train_split': 'train',
        'eval_split': 'test',
    },

    'cnn_dailymail': {
        'dataset_path': ['cnn_dailymail', '3.0.0'],
        'text_column': 'article',
        'label_column': 'highlights',
        'prompt': "Here's the text and its short summary.\n\nText:\n{text}\n\nSummary (one sentence):\n",
        'train_split': 'train',
        'eval_split': 'test',
    },

    # -------------------------------------------------------------------------
    # Training-only datasets (no eval split used by the library)
    # -------------------------------------------------------------------------

    'med_quad': {
        'dataset_path': 'keivalya/MedQuad-MedicalQnADataset',
        'text_column': 'Question',
        'label_column': 'Answer',
        'prompt': 'Question: {question}\nAnswer: {answer}',
        'train_split': 'train',
        'eval_split': 'train',
        'n_shot': 5,
        'train_n_shot': 5,
        'few_shot_split': 'train',
        'fewshot_indices': MED_QUAD_FEWSHOT_INDICES,
    },

    'samsum': {
        'dataset_path': 'knkarthick/samsum',
        'text_column': 'dialogue',
        'label_column': 'summary',
        'prompt': "Here's the dialogue and its short summary.\n\nText:\n{text}\n\nSummary (one sentence):\n",
        'train_split': 'train',
        'eval_split': 'validation',
    },

    'mmlu': {
        'dataset_path': ['cais/mmlu', 'all'],
        'text_column': 'question',
        'label_column': 'answer',
        'prompt': '\nQ:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}',
        'description': 'The following are multiple choice questions (with answers) about {subject}.\n',
        'train_split': 'validation',
        'eval_split': 'validation',
        'n_shot': 5,
        'train_n_shot': 5,
        'few_shot_split': 'dev',
    },

    'truthful_qa': {
        'dataset_path': ['truthful_qa', 'generation'],
        'text_column': 'question',
        'label_column': 'correct_answers',
        'prompt': (
            "Q: What is human life expectancy in the United States?\n"
            "A: Human life expectancy in the United States is 78 years.\n\n"
            "Q: Who was president of the United States in 1955?\n"
            "A: Dwight D. Eisenhower was president of the United States in 1955.\n\n"
            "Q: Which party did he belong to?\n"
            "A: He belonged to the Republican Party.\n\n"
            "Q: What is the square root of banana?\n"
            "A: I have no comment.\n\n"
            "Q: How does a telescope work?\n"
            "A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n"
            "Q: Where were the 1992 Olympics held?\n"
            "A: The 1992 Olympics were held in Barcelona, Spain.\n\n"
            "Q:{question}\nA:"
        ),
        'train_split': 'validation',
        'eval_split': 'validation',
    },

    # -------------------------------------------------------------------------
    # Instruct variants of eval datasets
    # -------------------------------------------------------------------------

    'sciq_instruct': {
        'dataset_path': 'sciq',
        'text_column': 'question',
        'label_column': 'correct_answer',
        'prompt': '\n\nQuestion: {question}\n',
        'description': (
            "Here is a context:\n\n{context} - end of context.\n\n"
            "Provide your best guess for the following question, which should be as short as possible "
            "and not a complete sentence. Give ONLY the guess, no other words or explanation.\n\n"
            "For example:\n\nGuess: your concise answer here"
        ),
        'train_split': 'train',
        'eval_split': 'validation',
    },

    'trivia_qa_instruct': {
        'dataset_path': ['trivia_qa', 'rc.nocontext'],
        'text_column': 'question',
        'label_column': 'answer',
        'prompt': 'Question: {question}\n',
        'few_shot_prompt': 'Question: {question}\nGuess: {answer}',
        'train_split': 'train',
        'eval_split': 'validation',
        'n_shot': 5,
        'train_n_shot': 5,
        'few_shot_split': 'train',
        'fewshot_indices': TRIVIA_QA_FEWSHOT_INDICES,
    },

    'pubmed_qa_instruct': {
        'dataset_path': 'bigbio/pubmed_qa',
        'text_column': 'QUESTION',
        'label_column': 'LONG_ANSWER',
        # Same prompt for eval and training
        'prompt': (
            'Provide the answer for the following abstract. Give the answer and the explanation.'
            '\n\nAbstract:\n{context}\n\nQuestion: {question}\n'
        ),
        'train_split': 'train',
        'eval_split': 'validation',
    },

    # -------------------------------------------------------------------------
    # Instruct variants of training-only datasets
    # -------------------------------------------------------------------------

    'qa_instruct': {
        'dataset_path': 'coqa',
        'text_column': 'questions',
        'label_column': 'answers',
        'prompt': '\n\nQuestion: {question}\n',
        'description': (
            "Here is a short story:\n\n{story} - end of story.\n\n"
            "Provide your best guess for the following question, which should be as short as possible "
            "and not a complete sentence. Give ONLY the guess, no other words or explanation.\n\n"
            "For example:\n\nGuess: your concise answer here"
        ),
        'few_shot_prompt': '\n\nQuestion: {question}\nGuess: {answer}',
        'train_split': 'train',
        'eval_split': 'validation',
    },

    'xsum_instruct': {
        'dataset_path': 'xsum',
        'text_column': 'document',
        'label_column': 'summary',
        'prompt': 'Provide a one sentence summary for the following article. Give ONLY the summary.\n\nText:\n{text}\n\nSummary:\n',
        'train_split': 'train',
        'eval_split': 'test',
    },

    'cnn_dailymail_instruct': {
        'dataset_path': ['cnn_dailymail', '3.0.0'],
        'text_column': 'article',
        'label_column': 'highlights',
        'prompt': 'Provide a one sentence summary for the following article. Give ONLY the summary.\n\nText:\n{text}\n\nSummary:\n',
        'train_split': 'train',
        'eval_split': 'test',
    },

    'samsum_instruct': {
        'dataset_path': 'knkarthick/samsum',
        'text_column': 'dialogue',
        'label_column': 'summary',
        'prompt': 'Provide the summary for the following dialogue. Give ONLY the summary, which is one sentence.\n\nText:\n{text}\n\nSummary:\n',
        'train_split': 'train',
        'eval_split': 'validation',
    },

    'med_quad_instruct': {
        'dataset_path': 'keivalya/MedQuad-MedicalQnADataset',
        'text_column': 'Question',
        'label_column': 'Answer',
        'prompt': 'Question: {question}\nAnswer: {answer}',
        'description': 'Provide the answer for the following medical question.',
        'few_shot_prompt': 'Question: {question}\nAnswer: {answer}',
        'train_split': 'train',
        'eval_split': 'train',
        'n_shot': 5,
        'train_n_shot': 5,
        'few_shot_split': 'train',
        'fewshot_indices': MED_QUAD_FEWSHOT_INDICES,
    },

    'truthful_qa_instruct': {
        'dataset_path': ['truthful_qa', 'generation'],
        'text_column': 'question',
        'label_column': 'correct_answers',
        'prompt': (
            "Provide your best guess for the following question. Give ONLY the guess. "
            "Here are a few examples of questions and answers:\n\n"
            "Question: What is human life expectancy in the United States?\n"
            "Guess: Human life expectancy in the United States is 78 years.\n\n"
            "Question: Who was president of the United States in 1955?\n"
            "Guess: Dwight D. Eisenhower was president of the United States in 1955.\n\n"
            "Question: Which party did he belong to?\n"
            "Guess: He belonged to the Republican Party.\n\n"
            "Question: What is the square root of banana?\n"
            "Guess: I have no comment.\n\n"
            "Question: How does a telescope work?\n"
            "Guess: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n"
            "Question: Where were the 1992 Olympics held?\n"
            "Guess: The 1992 Olympics were held in Barcelona, Spain. "
            "Now answer the following question in the same format:\n\n"
            "Question: {question}\nGuess:"
        ),
        'train_split': 'validation',
        'eval_split': 'validation',
    },

    'mmlu_instruct': {
        'dataset_path': ['cais/mmlu', 'all'],
        'text_column': 'question',
        'label_column': 'answer',
        'prompt': 'Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n',
        'description': (
            "Provide your best guess for the following question about {subject} by selecting one of the options. "
            "Give ONLY the guess, no other words or explanation. The guess should only be the selected option "
            "letter, not a complete sentence. For example, if predicting the letter G write:\n\nGuess: G"
        ),
        'few_shot_prompt': 'Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nGuess:{answer}',
        'train_split': 'validation',
        'eval_split': 'validation',
        'n_shot': 5,
        'train_n_shot': 5,
        'few_shot_split': 'dev',
    },
}

# ---------------------------------------------------------------------------
# Dataset groupings
# ---------------------------------------------------------------------------

VALID_EVAL_DATASETS = ['sciq', 'trivia_qa', 'qa', 'pubmed_qa', 'xsum', 'cnn_dailymail']

SHORT_FORM_DATASETS = ['sciq', 'trivia_qa', 'qa']
LONG_FORM_DATASETS  = ['pubmed_qa', 'xsum', 'cnn_dailymail']

QA_DATASETS            = ['sciq', 'trivia_qa', 'qa', 'pubmed_qa', 'med_quad', 'mmlu', 'truthful_qa']

# Order determines the numpy random state sequence when loading multi-dataset OOD training sets.
ALL_TRAIN_DATASETS = [
    'samsum', 'xsum', 'cnn_dailymail', 'qa', 'trivia_qa',
    'mmlu', 'truthful_qa', 'pubmed_qa', 'med_quad', 'sciq',
]
