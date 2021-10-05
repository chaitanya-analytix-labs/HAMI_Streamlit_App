import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim_models
import re
from nltk.tokenize import sent_tokenize
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from collections import Counter
# Sumy Summary Package
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

nlp = spacy.load("en_core_web_sm")


# Function for Sumy Summarization
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


# summary=df1['clean_text'].apply(lambda x: sumy_summarizer(x))


# Text Processing
def text_preprocessing(text):
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove new line characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Define tokenization function (sentence-level)
def sent_to_words(sentences):
    for sentence in sentences:
        yield sent_tokenize(str(sentence))


def lemmatization(
        texts,
        allowed_postags=[
            "NOUN",
            "PROPN",
            "ADJ",
            "VERB",
            "ADV",
            "NUM",
            "GPE",
            "DATE",
        ],
):
    """https://spacy.io/api/annotation"""
    texts_out = []

    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            " ".join(
                [
                    token.lemma_ if token.lemma_ not in ["-PRON-"] else ""
                    for token in doc
                    if token.pos_ in allowed_postags
                ]
            )
        )
    # lexeme.is_stop = True
    return texts_out


def sentiment_analyzer_scores(sentence) -> dict:
    """
    Compute the negative score, neutral score, positive score and compound score for each text sentences.

    :param: str sentence: sentence to analyse

    :return: all scores returned from VADER
    :rtype: dict

    """
    analyzer = SentimentIntensityAnalyzer()
    sentence = str(sentence)
    score = analyzer.polarity_scores(sentence)
    # print('sentence:', sentence)
    # print('neg:', score['neg'])
    # print('neu:', score['neu'])
    # print('pos:', score['pos'])
    # print('compound:', score['compound'])
    # return score['neg'], score['neu'], score['pos'], score['compound']
    return score['compound']


def get_ner_pos():
    valence = open("/Users/shaheemmuhammed/workspace/HAMI_Streamlit_App/code/vader_lexicon.txt", "r")

    # split the vader.txt file contents with respect to spaces
    lex_dict = {}
    for line in valence:
        (word, measure) = line.strip().split('\t')[0:2]
        lex_dict[word] = float(measure)

    # split those values that are less than '0' as negative and vice versa
    tokenized_words_neg = dict((k, v) for k, v in lex_dict.items() if v < 0)

    tokenized_words_pos = dict((k, v) for k, v in lex_dict.items() if v > 0)

    return tokenized_words_neg, tokenized_words_pos


tokenized_words_neg, tokenized_words_pos = get_ner_pos()


class AppData:
    df = pd.read_excel('/Users/shaheemmuhammed/workspace/HAMI_Streamlit_App/code/sample_TG.xlsx')
    df1 = df.rename(columns={'messages.date': 'date', 'messages.text': 'text', 'messages.from': 'from'})
    df1 = df1[['date', 'text', 'from']]
    df1['text'] = df1['text'].astype(str)
    df1['clean_text'] = df1['text'].apply(lambda text: text_preprocessing(text))
    corpus = df1['clean_text'].values.tolist()
    corpus_words = list(sent_to_words(corpus))

    # Execute lemmatization; keeping only items in allowed_postags
    corpus_lemmatized = lemmatization(
        corpus_words,
        allowed_postags=[
            "NOUN",
            "PROPN",
            "ADJ",
            "VERB",
            "ADV",
            "NUM",
            "ORG",
            "DATE",
        ],
    )

    # Prepare Stopwords
    stopwords = stopwords.words('english')
    stopwords.extend(['list', 'nan', 'link', 'type', 'thing', 'Haha', 'OK'])

    vectorizer = CountVectorizer(
        analyzer="word",
        min_df=5,
        stop_words=stopwords,
        ngram_range=(1, 2)
    )

    corpus_vectorized = vectorizer.fit_transform(corpus_lemmatized)

    search_params = {
        "n_components": [i for i in range(4, 6)],
    }

    # Initiate the Model
    lda = LatentDirichletAllocation(
        max_iter=100,  # default 10
        learning_method="batch",
        batch_size=32,
        learning_decay=0.7,
        random_state=42,
    )

    # Init Grid Search Class
    model = GridSearchCV(lda, cv=5, param_grid=search_params, n_jobs=-1, verbose=1)

    # Do the Grid Search
    model.fit(corpus_vectorized)

    # LDA Model
    best_lda_model = model.best_estimator_

    # pyLDAvis
    panel = pyLDAvis.sklearn.prepare(
        best_lda_model, corpus_vectorized, vectorizer, mds="tsne", R=50, sort_topics=False)
    pyLDAvis.save_html(panel, '/Users/shaheemmuhammed/workspace/HAMI_Streamlit_App/code/TG_py.html')

    # Creating new df with keywords count

    keywords = panel.topic_info[["Term", "Freq", "Total", "Category"]]

    keywords = keywords.rename(
        columns={
            "Term": "Keyword",
            "Freq": "Count in Topic",
            "Total": "Count Overall",
            "Category": "Topic",
        }
    )

    keywords["Topic"] = keywords["Topic"].map(
        lambda x: str(x).replace("Default", "Overall Transcript")
    )

    # Creating new dataframe with dominant topics and probability scores

    lda_output = best_lda_model.transform(corpus_vectorized)

    # column names
    topicnames = ["Topic" + str(i + 1) for i in range(best_lda_model.n_components)]

    # index names
    docnames = ["Doc" + str(i + 1) for i in range(len(corpus))]

    # Make the pandas dataframe
    TG_topics = pd.DataFrame(
        np.round(lda_output, 2), columns=topicnames, index=docnames
    )

    # Get dominant topic for each document
    dominant_topics = np.argmax(TG_topics.values, axis=1)
    TG_topics["DOMINANT_TOPIC"] = dominant_topics + 1

    lemma_text = pd.DataFrame(corpus_lemmatized, columns=["Lem_Text"])

    # Merging all 3 dataframes

    TG_topics.reset_index(inplace=True)

    full_results = df1.merge(lemma_text, left_index=True, right_index=True).merge(
        TG_topics, left_index=True, right_index=True)
    full_results.drop("index", axis=1, inplace=True)

    # exporting to excel
    full_results.to_excel("/Users/shaheemmuhammed/workspace/HAMI_Streamlit_App/code/TG_pyTopics.xlsx", index=False)
    keywords.to_excel("/Users/shaheemmuhammed/workspace/HAMI_Streamlit_App/code/TG_pykeywords.xlsx", index=False)

    # importing for calling with st.dataframe()
    full_results = pd.read_excel("/Users/shaheemmuhammed/workspace/HAMI_Streamlit_App/code/TG_pyTopics.xlsx")
    keywords = pd.read_excel("/Users/shaheemmuhammed/workspace/HAMI_Streamlit_App/code/TG_pykeywords.xlsx")

    df1['vader_comp'] = df1.apply(lambda x: sentiment_analyzer_scores(x.clean_text), axis=1, result_type='expand')

    df1['vader_prediction'] = df1.apply(
        lambda x: 'positive' if x.vader_comp > 0 else ('negative' if x.vader_comp < 0 else 'neutral'), axis=1)

    df1['neg_keywords'] = df1.apply(
        lambda row: [word for word in word_tokenize(row['clean_text']) if word.lower() in tokenized_words_neg], axis=1)
    df1['pos_keywords'] = df1.apply(
        lambda row: [word for word in word_tokenize(row['clean_text']) if word.lower() in tokenized_words_pos], axis=1)

    df1.to_excel("/Users/shaheemmuhammed/workspace/HAMI_Streamlit_App/code/Tg_sentiments.xlsx")

    ####################################################
    # APPLYING the NER FROM SPACY
    ####################################################

    df1['NER'] = df1['clean_text'].apply(lambda x: nlp(x))

    tokens = nlp(''.join(str(df1.clean_text.tolist())))
    items = [x.text for x in tokens.ents]
    # Counter(items).most_common(20)

    # Names of persons
    person_list = []
    for ent in tokens.ents:
        if ent.label_ == 'PERSON':
            person_list.append(ent.text)

    person_counts = Counter(person_list).most_common(20)
    df_person = pd.DataFrame(person_counts, columns=['text', 'count'])

    # Nationalities, religious and political groups
    norp_list = []
    for ent in tokens.ents:
        if ent.label_ == 'NORP':
            norp_list.append(ent.text)

    norp_counts = Counter(norp_list).most_common(20)
    df_norp = pd.DataFrame(norp_counts, columns=['text', 'count'])

    # Companies Agencies institutions

    org_list = []
    for ent in tokens.ents:
        if ent.label_ == 'ORG':
            org_list.append(ent.text)

    org_counts = Counter(org_list).most_common(20)
    df_org = pd.DataFrame(org_counts, columns=['text', 'count'])

    # Objects,vehicles,foods(not services)

    prod_list = []
    for ent in tokens.ents:
        if ent.label_ == 'PRODUCT':
            prod_list.append(ent.text)

    prod_counts = Counter(prod_list).most_common(20)
    df_prod = pd.DataFrame(prod_counts, columns=['text', 'count'])

    # Countres, cities and states
    gpe_list = []
    for ent in tokens.ents:
        if ent.label_ == 'GPE':
            gpe_list.append(ent.text)

    gpe_counts = Counter(gpe_list).most_common(20)
    df_gpe = pd.DataFrame(gpe_counts, columns=['text', 'count'])

    sumy = sumy_summarizer(corpus)
