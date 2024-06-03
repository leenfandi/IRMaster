# IRMaster
# System Services
# Project Overview

## Dataset Description

### Antique
- **Documents**: 404K (text, id)
- **Queries**: 2.4K (id, query)
- **Qrels**: 27K (query_id, doc_id, relevance, iteration)

### Wikir/en1k
- **Documents**: 370K (text, id)
- **Queries**: 48K (id, query)
- **Qrels**: 48K (query_id, doc_id, relevance, iteration)

## Preprocessing Steps

1. **Expand Words**: Use `contractions` to expand abbreviations.
2. **Tokenize Words**: Use `word_tokenize` to split text into tokens.
3. **Remove Punctuation**
4. **Normalization**:
   - Standardize dates using `dateutil.parser`.
   - Convert numbers to words using `num2word`.
   - Expand abbreviations (terms, countries, states) using a predefined dictionary.
   - Standardize American and British spellings using a predefined dictionary.
   - Convert Unicode to ASCII using `anyascii`.
5. **Stemming**: Remove prefixes and suffixes using `PorterStemmer`.
6. **Part of Speech Tagging**: Identify grammatical categories for use in lemmatization.
7. **Lemmatization**: Convert words to their root forms using `WordNetLemmatizer`.
8. **Stop Words Removal**

## Data Representation & Indexing

- **TF-IDF**: Using `sklearn` to create a vector space model.
  - **Parameters**:
    - `sublinear_tf=True`
    - `norm='l2'`
    - `max_df=0.9`

## Query Processing

- Apply the same preprocessing steps to the query.

## Query Matching & Ranking

- Represent the query as a vector.
- Use `cosine_similarity` to match the query with document vectors and rank them.

## Evaluation Metrics

- **Precision at k**: Precision of the top k retrieved documents.
- **Recall at k**: Recall of the top k retrieved documents.
- **Mean Average Precision (MAP)**
- **Mean Reciprocal Rank (MRR)**

## Results

### Antique Dataset
- **MAP**: 0.14731879736223874
- **MRR**: 0.08120193050879908
- **mean_precision_10**:0.0304204451772465
- **mean_recal**:0.11765890734283808

### Antique Dataset after clustering
- **MAP**: 0.5787161922272746
- **MRR**:  0.2408775958858399
- **mean_precision_10**:0.07061005770816159
- **mean_recal**:, 0.11765890734283808

### Wikir/en1k Dataset
- **MAP**: 0.5404
- **MRR**: 0.6094

## Additional Features

- **Multi-Lingual Retrieval System**: Uses Google Translator to handle Arabic queries and return results in the same language.

1. **Voice Search and Retrieval**: The system supports searching and retrieving results through voice input.
2. **Dataset Switching**: The ability to switch datasets through the user interface.
3. **Language Switching**: The ability to switch languages through the user interface and to search by either text or voice in both Arabic and English, with results returned in the same language as the search was conducted.

# System Structure Description

1. The process starts with reading from a file, and the result of the reading is stored in a variable.
2. The text in this variable undergoes processing, followed by stemming to improve the accuracy of the results.
3. Lemmatization is then applied.
4. Different expressions are found and appropriate replacements are made using a dictionary.
5. The tf-idf matrix is prepared, representing documents as vectors, and the query is also represented as a vector to calculate the similarity between them.

