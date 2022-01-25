# Hide Rerun Menu options for users
hide_menu = """
<style>
#MainMenu {
  visibility:hidden;
}
footer {
    visibility:visible;
}
footer:before{
    content:'Copyright.@.2022:AnalytixOnline';
    display:block;
    position:Relative;
    color:rgba(38, 39, 48, 0.4);
    padding:5px;
    top:3px;
}
<style>
"""

# PYNGROK
from contextlib import redirect_stderr
import joblib

# reddit

from datetime import datetime
import praw
import pandas as pd
import numpy as np
import streamlit as st

import time
# streamlit-tags
from streamlit_tags import st_tags  # pip install streamlit_tags==1.0.6

# lottie-Animation
from streamlit_lottie import st_lottie
import requests

# Read Docx files

import collections

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords

# Importing spacy for Lemmatization
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 14500000  # or even higher
from collections import Counter

# sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# import streamlit
import streamlit as st

st.set_page_config(layout="wide")
import streamlit.components.v1 as components
import io
import os
import base64

# Tools required for plotting and wordcloud
import seaborn as sns
from wordcloud import WordCloud

################################
# Text matching
################################
from sentence_transformers import SentenceTransformer
from scipy import spatial
import json
import torch
from nltk import tokenize

###############################
# Drill down Analysis
###############################
import plotly.express as px
import plotly.graph_objects as go

###############################
# Twitter API packages
###############################
import tweepy as tw
from tweepy import API
from tweepy import Cursor
from tweepy.streaming import Stream
from tweepy import OAuthHandler
from tweepy import Stream

from textblob import TextBlob

import twitter_credentials
import reddit_credentials

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# Regex string for extract specific chartacters/ numbers
regex_str = r'\d{5,8}'

# Read Docx files

from PyPDF2 import PdfFileReader


#################################
# Lottie Animation
#################################
# Load the animation from local folder
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# Load animation directly from website
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json


proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# import model file

#model = SentenceTransformer(model_dir + '/paraphrase-distilroberta-base-v1')
model = SentenceTransformer('/Volumes/GoogleDrive-110033092045285714630/My Drive/HAMI/Production/HAMI_Streamlit_App_v3.0/HAMI_Streamlit_App/HAMI_Streamlit_App/model_for_similarity/paraphrase-distilroberta-base-v1')
lottie_home = load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_3ezlslmp.json")

lottie_home2 = load_lottiefile(proj_dir + "/Hami_home.json")

lottie_home3 = load_lottiefile(proj_dir + "/Hami_hello.json")


##################################
# PDF Reader
##################################
def readpdf(file):
    # Load PDF into pyPDF
    pdf = PdfFileReader(file)
    # Extract the first page
    count = PdfFileReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdf.getPage(i)
        page_text = page.extractText()
        all_page_text += page_text
    return all_page_text


# %matplotlib inline

import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim_models

st.set_option('deprecation.showPyplotGlobalUse', False)

# pyLDAvis.enable_notebook()


# Load Text Cleaning Pkgs
import neattext.functions as nfx


# @st.cache
# Text Preprocessing
def text_preprocessing(text):
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove emails
    text = re.sub(r'(\W|^)[\w.\-]{0,25}@(yahoo|hotmail|gmail)\.com(\W|$)', ' ', text)
    # Remove new line characters
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters, numbers, punctuations, etc.
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove duplicate spaces
    # text = re.sub(r'\s+', ' ', text)
    # Remove the text present in between paranthases
    # text = re.sub(r'\([^)]*\)', '', text)
    # Remove the text present in between braces
    # text = re.sub(r'\{[^)]*\}', '', text)
    # Remove the text present in between <>
    # text = re.sub(r'<[^)]*>', '', text)
    # Remove the text present in between []
    # text = re.sub(r'\[[^)]*\]', '', text)

    return text


# @st.cache
def show_ents(doc):
    if doc.ents():
        for ent in doc.ents:
            print(ent.text + '-' + ent.label_ + '-' + str(spacy.explain(ent.label_)))

    else:
        print('No named entities found')


# word cloud
# Define tokenization function (word-level)
def sent_to_words(sentence):
    for word in sentence:
        yield (word_tokenize(str(word)))


# Prepare Stopwords

stopwords = stopwords.words('english')
stopwords.extend(
    ['list', 'listtype', 'listlisttype', 'nan', 'link', 'type', 'thing', 'Haha', 'OK', "'lol'", 'nil', "nil'", "https",
     "www", "think", "like", "text", "lol", "no'", "like'", "text", "com", "2021", "covid", "19", "vaccine", "'"])


#############################
# Word count across timeline
#############################


#############################
# TOPIC MODELLING
#############################
# @st.cache
# Define tokenization function (sentence-level) 
def para_to_sents(paragraph):
    for sentence in paragraph:
        yield (sent_tokenize(str(sentence)))


# @st.cache
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
    return texts_out
    lexeme.is_stop = True


#############################
# SENTIMENT PREDICTION
#############################

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler

analyzer = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(sentence) -> dict:
    """
    Compute the negative score, neutral score, positive score and compound score for each text sentences.

    :param: str sentence: sentence to analyse

    :return: all scores returned from VADER
    :rtype: dict

    """
    sentence = str(sentence)
    score = analyzer.polarity_scores(sentence)
    # print('sentence:', sentence)
    # print('neg:', score['neg'])
    # print('neu:', score['neu'])
    # print('pos:', score['pos'])
    # print('compound:', score['compound'])
    # return score['neg'], score['neu'], score['pos'], score['compound']
    return score['compound']


######################
# SUMMARIZATION
######################

# Sumy Summary Package
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# Function for Sumy Summarization
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


# T5 Summarization model
# model.load_model("t5","/content/outputs/simplet5-epoch-4-train-loss-0.6352", use_gpu=True)

############################################
# sql database management to store passwords
#############################################
import hashlib


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


import sqlite3

conn = sqlite3.connect('data.db')
c = conn.cursor()


def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS usertable(username TEXT,password TEXT,email TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO usertable(username,password) VALUES (?,?)', (username, password))
    conn.commit()


def login_user(username, password):
    c.execute('SELECT * from usertable WHERE username = ? AND password = ?', (username, password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM usertable')
    data = c.fetchall()
    return data


# select columns
def select_cols(self):
    if self.df is not None:
        try:
            df_columns = list(self.df.columns)
            label = 'Select Columns'
            self.usecols = st.multiselect(label=label, default=df_columns, options=df_columns)
            self.df = self.df[self.usecols]
        except:
            st.write('Column Select Error')


def main():
    # Top Headers
    st.sidebar.image(proj_dir + '/HAMI LOGO.png')
    st.sidebar.title("ANALYTIX ONLINE")
    st.sidebar.subheader("AI Solutions, Fine-Tuned to You")

    # Hide Menu and edit Footer
    st.markdown(hide_menu, unsafe_allow_html=True)

    # streamlit commands
    st.title("HAMI Data Analysis App")
    menu = ["Home", "Login", "Create Account"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st_lottie(
            lottie_home2,
            speed=1,
            reverse=False,
            loop=True,
            quality="high",
            key=None,
            width=1150,
            height=850,
        )

    elif choice == "Login":
        st.subheader("Login Section")
        col1, col2, col3, = st.columns(3)
        with col2:
            st_lottie(
                lottie_home3,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                key=None
            )

        username = st.sidebar.text_input("User Name")
        # email=st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password", type='password')
        global stopwords
        if st.sidebar.checkbox("Login"):
            # if password=='admin':
            create_usertable()
            result = login_user(username, password)

            if result:
                st.success("Logged In as {}".format(username))

                # Header/Subheader
                st.header("Analysis")

                # Uploader

                dff = st.file_uploader("Choose a file to Upload",
                                       type=["xlsx", "csv", "json", "docx", "xls"])

                global data  # To make it available outside the if statement

                if dff is not None:
                    try:
                        # un-comment for accessing by sheet names
                        # sheets=["FG1 HQ Exec","FG2","FG3","FG4","FG5","FG6 Ops Manager"]
                        data = pd.read_excel(dff)  # ,sheet_name=sheet_name)
                        col_select = st.selectbox(
                            label="Select the column for analysis",
                            options=data.columns
                        )
                        # sheets=get_sheet_details(data)
                        # sheet_name=st.selectbox('select the sheet:',options=sheets)

                        # Remove empty rows
                        # data=data.dropna()
                        st.dataframe(data)

                        # DATA_COLUMN
                        data[col_select] = data[col_select].astype(str)

                        # data['clean_text']=data.loc[data[col_select].str.len()>2]
                        remove_list = ['list', 'nan', 'joined', 'forward']
                        # data=data.loc[~data['clean_text'].str.lower().str.contains('|'.join(remove_list),na=False)]
                        data['clean_text'] = data[col_select].astype(str)
                        # User handles
                        data['clean_text'] = data['clean_text'].apply(nfx.remove_userhandles)

                        # User html_tags
                        data['clean_text'] = data['clean_text'].apply(nfx.remove_html_tags)

                        # Remove custom Stopwords
                        data['clean_text'] = data['clean_text'].apply(nfx.remove_stopwords)
                        # Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
                        data['clean_text'] = data['clean_text'].apply(
                            lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

                        # Remove URLs
                        data['clean_text'] = data['clean_text'].apply(nfx.remove_urls)

                        data['clean_text'] = data['clean_text'].apply(lambda text: text_preprocessing(text))

                        filtered_data = data['clean_text']
                        topic_corpus = filtered_data.astype(str)
                        topic_text = topic_corpus.values.tolist()
                        topic_corpus_words = list(para_to_sents(topic_text))




                    except ValueError as error:
                        data = pd.read_csv(dff)
                        col_select = st.selectbox(
                            label="Select the column for analysis",
                            options=data.columns
                        )

                        # data=data.dropna()
                        st.dataframe(data)

                        # DATA_COLUMN
                        data[col_select] = data[col_select].astype(str)

                        # data['clean_text']=data.loc[data[col_select].str.len()>2]
                        remove_list = ['list', 'nan', 'joined', 'forward']
                        # data=data.loc[~data['clean_text'].str.lower().str.contains('|'.join(remove_list),na=False)]
                        data['clean_text'] = data[col_select].astype(str)
                        # User handles
                        data['clean_text'] = data['clean_text'].apply(nfx.remove_userhandles)

                        # User html_tags
                        data['clean_text'] = data['clean_text'].apply(nfx.remove_html_tags)

                        # Stopwords
                        data['clean_text'] = data['clean_text'].apply(nfx.remove_stopwords)
                        # Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
                        # data['clean_text'] = data['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

                        # Remove URLs
                        data['clean_text'] = data['clean_text'].apply(nfx.remove_urls)

                        # Text Preprocessing
                        data['clean_text'] = data['clean_text'].apply(lambda text: text_preprocessing(text))

                        filtered_data = data['clean_text']
                        topic_corpus = filtered_data.astype(str)
                        topic_text = topic_corpus.values.tolist()
                        topic_corpus_words = list(para_to_sents(topic_text))

                    # File details
                    file_details = {"Filename": dff.name,
                                    "FileType": dff.type, "FileSize": dff.size}
                    st.write(file_details)

                else:
                    st.warning("you need to upload a file to start the analysis")

                if st.checkbox("Textual Data"):

                    ################################################
                    # Streamlit Deck
                    ################################################
                    # Submit button
                    with st.form(key="form1"):
                        # SelectBox
                        options = st.radio("Select the analysis task",
                                           ["Topic Modelling", "Entity analysis", "Sentiment Analysis",
                                            "Emotion Analysis", "Text Summarization",
                                            "Time Series Analysis", "Text Similarity", "Twitter Sentiment Analysis","Reddit text analysis"],help='Select the task and click submit')
                        submit = st.form_submit_button(label="Submit")

                        #########################
                        # TOPIC MODELLING
                        #########################

                        keywords = st_tags('Enter custom Stopwords:', 'Press enter to add more', ['hello'])

                        stopwords.extend(keywords)
                        
                       

                        if options == "Topic Modelling" and submit:
                                
                            st.subheader("Topic Modelling")
                            st.subheader("""Topic modeling gives us a way to infer the latent structure behind a collection of texts.
                            Words/phrases are assigned to Topics based on their association with other words/phrases in the input collection.
                            Topics are then used to infer the underlying semantic structure present in each cluster.
                            
                            1. The area of the circle represents the importance of each topic over the entire corpus.
                                    The bigger the circle the higher the importance of that particular topic.
	                        2. The distance between the center of each of these circles indicate the similarity between the topics.
                                    The histogram on the right indicates top 30 relevant terms among each topic.""")                       

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.subheader("""1. Enter the Minimum required occurences of a word to be counted for model""")
                                minimum_df = st.slider('slide to set min_df', min_value=1, max_value=8,
                                                       help="Minimum required occurences of a word")
                            with col2:
                                st.subheader("""2. Enter the Number of topics to be formed by model""")
                                collect_numbers = lambda x: [int(i) for i in re.split("[^0-9]", x) if i != ""]
                                number_of_topics = st.text_input('Enter number of topics(minimum is 2)', "2")
                                ticks = (collect_numbers(number_of_topics))
                            with col3:
                                st.subheader("""3. Assign the number for keywords pairs i.e.Bi-gram, tri-gram,...n-gram """)
                                n_grams = st.slider('select a range of n-grams', 1, 5, (1, 2),
                                                    help="Assign the number ngram for keywords/phrases i.e.Bi-gram, tri-gram,...n-gram")


                            # elif options == "Topic Modelling" and submit:

                            topic_corpus_lemmatized = lemmatization(topic_corpus_words,
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

                            topic_vectorizer = CountVectorizer(
                                analyzer="word",
                                min_df=minimum_df,
                                stop_words=stopwords,
                                ngram_range=n_grams
                            )

                            topic_corpus_vectorized = topic_vectorizer.fit_transform(topic_corpus_lemmatized)

                            search_params = {
                                "n_components": ticks,
                            }

                            # Initiate the Model
                            topic_lda = LatentDirichletAllocation(
                                max_iter=100,  # default 10
                                learning_method="batch",
                                batch_size=32,
                                learning_decay=0.7,
                                random_state=42, )

                            # Init Grid Search Class
                            topic_model = GridSearchCV(topic_lda, cv=5, param_grid=search_params, n_jobs=-1, verbose=1)

                            # Do the Grid Search
                            topic_model.fit(topic_corpus_vectorized)

                            # LDA Model
                            topic_best_lda_model = topic_model.best_estimator_

                            # pyLDAvis
                            topic_panel = pyLDAvis.sklearn.prepare(
                                topic_best_lda_model, topic_corpus_vectorized, topic_vectorizer, mds="tsne", R=50,
                                sort_topics=False)
                            pyLDAvis.save_html(topic_panel, proj_dir + '/topic_panel.html')

                            # Creating new df with keywords count

                            topic_keywords = topic_panel.topic_info[["Term", "Freq", "Total", "Category"]]

                            topic_keywords = topic_keywords.rename(
                                columns={
                                    "Term": "Keyword",
                                    "Freq": "Count in Topic",
                                    "Total": "Count Overall",
                                    "Category": "Topic",
                                }
                            )

                            topic_keywords["Topic"] = topic_keywords["Topic"].map(
                                lambda x: str(x).replace("Default", "Overall Transcript")
                            )

                            # Creating new dataframe with dominant topics and probability scores

                            topic_lda_output = topic_best_lda_model.transform(topic_corpus_vectorized)

                            # column names
                            Topicnames = ["Topic" + str(i + 1) for i in range(topic_best_lda_model.n_components)]

                            # index names
                            Docnames = ["Doc" + str(i + 1) for i in range(len(topic_corpus))]

                            # Make the pandas dataframe
                            Topics = pd.DataFrame(
                                np.round(topic_lda_output, 2), columns=Topicnames, index=Docnames
                            )

                            # Get dominant topic for each document
                            Dominant_topics = np.argmax(Topics.values, axis=1)
                            Topics["DOMINANT_TOPIC"] = Dominant_topics + 1

                            topic_lemma_text = pd.DataFrame(topic_corpus_lemmatized, columns=["Lem_Text"])

                            # Merging all 3 dataframes

                            Topics.reset_index(inplace=True)

                            topic_full_results = data.merge(topic_lemma_text, left_index=True, right_index=True).merge(
                                Topics, left_index=True, right_index=True)
                            topic_full_results.drop("index", axis=1, inplace=True)

                            # exporting to excel
                            topic_full_results.to_excel(proj_dir + "/Topics.xlsx", index=False)
                            topic_keywords.to_excel(proj_dir + "/Keywords.xlsx", index=False)

                            # importing for calling with st.dataframe()
                            topic_full_results = pd.read_excel(proj_dir + "/Topics.xlsx")
                            topic_keywords = pd.read_excel(proj_dir + "/Keywords.xlsx")

                            # st.plotly_chart(topic_panel)
                            html_string = pyLDAvis.prepared_data_to_html(topic_panel)
                            from streamlit import components
                            components.v1.html(html_string, width=1300, height=900, scrolling=False)

                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Dominant Topic Output**")
                                fr = pd.read_excel(proj_dir + '/Topics.xlsx')
                                st.dataframe(fr)
                                
                                # Export fr to excel
                                towrite = io.BytesIO()
                                downloaded_file = fr.to_excel(towrite, encoding='utf-8', index=False, header=True)
                                towrite.seek(0)  # reset pointer
                                b64 = base64.b64encode(towrite.read()).decode()  # some strings
                                linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Dominant_Topic.xlsx">Download Dominant_Topic file</a>'
                                st.markdown(linko, unsafe_allow_html=True)

                            with col2:
                                st.write("**Keyword List**")
                                kw = pd.read_excel(proj_dir + '/Keywords.xlsx')
                                st.dataframe(kw)
                                # Export kw to excel
                                towrite = io.BytesIO()
                                downloaded_file = kw.to_excel(towrite, encoding='utf-8', index=False, header=True)
                                towrite.seek(0)  # reset pointer
                                b64 = base64.b64encode(towrite.read()).decode()  # some strings
                                linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Keyword_list.xlsx">Download Keyword_list file</a>'
                                st.markdown(linko, unsafe_allow_html=True)

                            ####################################################
                            # TEXT SEARCHING
                            ####################################################


                        elif dff is not None and options == "Text Similarity" and submit:

                            st.subheader("""Text Similarity
                            Measures the similarity between two texts by transforming the sentences into vectors and then computing the cosine similarity between them.""")

                            ####################################################
                            # Specify a number form 1 ~ 10, higher is more strict
                            StrictLevel = st.slider("Strictness Level", 1, 10, 1)
                            # Confirmation target sentence
                            st.header("Enter a sentence or keyword to search for")
                            TargetSentence = st.text_input('Enter sentence to be searched')
                            # Confirmation key words in agent's sentences
                            #TargetKeyWords = st_tags('Enter target keywords:', 'Press enter to add', ['vaccination'])
                            # Confirmation response key words in customer's sentences       

                            ####################################################
                            if TargetSentence is not None:# or TargetKeyWords != '':

                                try:
                                    ####################################################
                                    # Sentence Similarity
                                    ####################################################
                                    def _check_word_in_sentence(words: list, sentence: str) -> bool:
                                        # Tokenize the sentence into tokens
                                        tokens = [word.lower() for word in tokenize.word_tokenize(sentence)]
                                        for word in words:
                                            if word in tokens:
                                                return True
                                        return False

                                    sentence_list = []
                                    for index, row in data.iterrows():
                                        sentences = [sent for sent in nltk.sent_tokenize(row[col_select])]
                                        for sentence in sentences:
                                            sentence_list.append([index, sentence])
                                    sentence_df = pd.DataFrame(sentence_list, columns=['index', 'text'])

                                    # Get the sentence embedding
                                    sentence_df['embedding'] = sentence_df['text'].apply(
                                        lambda text: model.encode(text))

                                    # Get the sentence embedding for target sentences
                                    confirmation_sentence = TargetSentence

                                    confirmation_embedding = model.encode(confirmation_sentence)

                                    # Calculate the similarity
                                    sentence_df['confirmation_sim'] = sentence_df['embedding'].apply(
                                        lambda x: 1 - spatial.distance.cosine(x, confirmation_embedding))

                                    # Calculate the rank of the similarity
                                    sentence_df['confirmation_order'] = sentence_df['confirmation_sim'].rank(
                                        method='first', ascending=False)

                                    ####################################################
                                    # keyword similarity
                                    ####################################################
                                    #temp_dict = {}

                                    #for index, row in sentence_df.iterrows():
                                    #    if (row['confirmation_order'] <= 11 - StrictLevel) and \
                                    #            (row['confirmation_sim'] > 0.35) or \
                                    #            _check_word_in_sentence(TargetSentence, row['text']):
                                    #        st.write('Matched rows with Sentence:', int(row['index']), row['text'])
                                    #    temp_dict = {
                                    #        'index': int(row['index']),
                                    #        'text': row['text']
                                    #    }
                                    # for keyword in TargetKeyWords:
                                    #for index, row in sentence_df.iterrows():
                                    #    if (_check_word_in_sentence(TargetSentence, row['text'])):
                                    #        st.write('Matched rows with keywords:', int(row['index']), row['text'])

                                    ####################################################

                                    st.write("**Text Similarity results**")
                                    df = data
                                    # df = df.dropna()

                                    st.write('Spreadsheet with similarity score for each row')
                                    st.dataframe(sentence_df)  # ,sorted='confirmation_sim')

                                    # Export to excel
                                    towrite = io.BytesIO()
                                    downloaded_file = sentence_df.to_excel(towrite, encoding='utf-8', index=False,
                                                                           header=True)
                                    towrite.seek(0)  # reset pointer
                                    b64 = base64.b64encode(towrite.read()).decode()  # some strings
                                    linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="similarity_score.xlsx">Download Similarity score file</a>'
                                    st.markdown(linko, unsafe_allow_html=True)

                                except:
                                    st.warning("Please enter a sentence and keyword for finding a match")




                        elif options == "Entity analysis" and submit:
                            st.subheader("""Entity analysis
                            Extracts entities from a text and returns the entities with their types.""")

                            ####################################################
                            # ENTITY RECOGNITION : APPLYING the NER FROM SPACY
                            ####################################################
                            st.write("**Select the Entity from dropdown to see results**")

                            # data['NER']=data['clean_text'].apply(lambda x: nlp(x))

                            tokens = nlp(''.join(str(data.clean_text.tolist())))
                            items = [x.text for x in tokens.ents]
                            # Counter(items).most_common(20)

                            select_plot = st.selectbox("Entity types identified", ("Person : People including fictional"
                                                                                   ,
                                                                                   "NORP : Nationalities or religious or political groups"
                                                                                   ,
                                                                                   "ORG : Companies, agencies, institutions."
                                                                                   , "GPE : Countries, cities, states"
                                                                                   ,
                                                                                   "PRODUCT : Objects, vehicles, foods"
                                                                                   ,
                                                                                   "EVENT : Named hurricanes, battles, wars, sports events"
                                                                                   ,
                                                                                   "LAW : Named documents made into laws"
                                                                                   ,
                                                                                   "WORK_OF_ART: Titles of books, songs"
                                                                                   ))

                            if select_plot == "Person : People including fictional":

                                # Names of persons

                                person_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'PERSON' and ent.text not in stopwords:
                                        person_list.append(ent.text)
                                person_counts = Counter(person_list).most_common(20)
                                df_person = pd.DataFrame(person_counts, columns=['text', 'count'])

                                st.pyplot(
                                    df_person.plot.barh(x='text', y='count', title="Names of persons", color="#80852c",
                                                        figsize=(10, 8)).invert_yaxis())

                            elif select_plot == "NORP : Nationalities or religious or political groups":

                                # Nationalities, religious and political groups

                                norp_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'NORP' and ent.text not in stopwords:
                                        norp_list.append(ent.text)
                                norp_counts = Counter(norp_list).most_common(20)
                                df_norp = pd.DataFrame(norp_counts, columns=['text', 'count'])

                                st.pyplot(df_norp.plot.barh(x='text', y='count', color="#e0c295",
                                                            title="Nationalities, religious and political groups",
                                                            figsize=(10, 8)).invert_yaxis())

                            elif select_plot == "ORG : Companies, agencies, institutions.":

                                # Companies Agencies institutions

                                org_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'ORG' and ent.text not in stopwords:
                                        org_list.append(ent.text)

                                org_counts = Counter(org_list).most_common(20)
                                df_org = pd.DataFrame(org_counts, columns=['text', 'count'])

                                st.pyplot(df_org.plot.barh(x='text', y='count', color="#cdbb69",
                                                           title="Companies Agencies institutions",
                                                           figsize=(5, 4)).invert_yaxis())

                            elif select_plot == "PRODUCT : Objects, vehicles, foods":

                                # Objects,vehicles,foods(not services)

                                prod_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'PRODUCT' and ent.text not in stopwords:
                                        prod_list.append(ent.text)

                                prod_counts = Counter(prod_list).most_common(20)
                                df_prod = pd.DataFrame(prod_counts, columns=['text', 'count'])

                                st.pyplot(df_prod.plot.barh(x='text', y='count', color="#eca349",
                                                            title="Objects,vehicles,foods",
                                                            figsize=(5, 4)).invert_yaxis())

                            elif select_plot == "EVENT : Named hurricanes, battles, wars, sports events":

                                # EVENT: Named hurricanes, battles, wars, sports events,
                                event_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'EVENT' and ent.text not in stopwords:
                                        event_list.append(ent.text)

                                event_counts = Counter(event_list).most_common(20)
                                df_event = pd.DataFrame(event_counts, columns=['text', 'count'])

                                st.pyplot(df_event.plot.barh(x='text', y='count', color="#e08a31",
                                                             title="Named hurricanes, battles, wars, sports events",
                                                             figsize=(5, 4)).invert_yaxis())

                            elif select_plot == "LAW : Named documents made into laws":

                                # LAW: Named documents made into laws
                                law_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'LAW' and ent.text not in stopwords:
                                        law_list.append(ent.text)

                                law_counts = Counter(law_list).most_common(20)
                                df_law = pd.DataFrame(law_counts, columns=['text', 'count'])

                                st.pyplot(df_law.plot.barh(x='text', y='count', color="#80852c",
                                                           title="Named documents made into laws",
                                                           figsize=(5, 4)).invert_yaxis())

                            elif select_plot == "WORK_OF_ART: Titles of books, songs":

                                # WORK_OF_ART: Titles of books, songs
                                workofart_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'WORK_OF_ART' and ent.text not in stopwords:
                                        workofart_list.append(ent.text)

                                workofart_counts = Counter(workofart_list).most_common(20)
                                df_workofart = pd.DataFrame(workofart_counts, columns=['text', 'count'])

                                st.pyplot(df_workofart.plot.barh(x='text', y='count', color="#e0c295",
                                                                 title="Named documents made into laws",
                                                                 figsize=(5, 4)).invert_yaxis())



                            elif select_plot == "GPE : Countries, cities, states":

                                # Countres, cities and states

                                gpe_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'GPE' and ent.text not in stopwords:
                                        gpe_list.append(ent.text)

                                gpe_counts = Counter(gpe_list).most_common(20)
                                df_gpe = pd.DataFrame(gpe_counts, columns=['text', 'count'])

                                st.pyplot(df_gpe.plot.barh(x='text', y='count', color="#e08a31",
                                                           title="Countres, cities and states",
                                                           figsize=(10, 8)).invert_yaxis())

                            else:
                                st.stop()

                            #########################
                            # Text Summarization
                            #########################    

                        elif options == "Text Summarization" and submit:
                            st.subheader("""Text Summarization
                            Unsupervised Machine Learning Algorithm to summarize text based on Graph-based Centrality Scoring of Sentences.
                            The sentences “recommend” other similar sentences to the user. Thus, if one sentence is very similar to many others, 
                            it will likely be a sentence of great importance
                            """)

                            # Lexrank

                            ##Uncomment if wanted to run the summarization row wise
                            # summary=data['clean_text'].apply(lambda x: sumy_summarizer(x))
                            # DATA_COLUMN
                            data[col_select] = data[col_select].astype(str)
                            data['clean_text'] = data[col_select].apply(lambda text: text_preprocessing(text))
                            filtered_data = data[col_select]
                            topic_text = filtered_data.astype(str)
                            topic_corpus = topic_text.values.tolist()
                            sumy = sumy_summarizer(topic_corpus)
                            st.write('Lex-Rank Summarizer')
                            st.success(sumy)

                            # Luhn
                            from sumy.summarizers.luhn import LuhnSummarizer
                            summarizer_luhn = LuhnSummarizer()
                            parser = PlaintextParser.from_string(topic_corpus, Tokenizer("english"))
                            summary_1 = summarizer_luhn(parser.document, 2)
                            st.write('LUHN Summarizer', help="heurestic method")
                            for sentence in summary_1:
                                st.success(sentence)

                            # LSA
                            from sumy.summarizers.lsa import LsaSummarizer
                            summarizer_lsa = LsaSummarizer()
                            summary_2 = summarizer_lsa(parser.document, 2)
                            st.write('LSA Summarizer', help="Latent Semantic Analysis")
                            for sentence in summary_2:
                                st.success(sentence)

                            # Using stopwords
                            ## Alternative Method using stopwords
                            from sumy.nlp.stemmers import Stemmer
                            from sumy.utils import get_stop_words
                            summarizer_lsa2 = LsaSummarizer()
                            summarizer_lsa2 = LsaSummarizer(Stemmer("english"))
                            summarizer_lsa2.stop_words = get_stop_words("english")
                            st.write('LSA2 Summarizer', help="LSA using customized stopwords")
                            for sentence in summarizer_lsa2(parser.document, 2):
                                st.success(sentence)




                        elif options == "Time Series Analysis" and submit:

                            st.subheader("""Time Series Analysis
                            Time Series Analysis is a powerful tool for analyzing time series data.
                            """)

                            # Create a function to get the subjectivity of the text
                            # def sentiment_analysis(data):
                            def getSubjectivity(text):
                                return TextBlob(text).sentiment.subjectivity

                                # Create a function to get the polarity

                            def getPolarity(text):
                                return TextBlob(text).sentiment.polarity

                            # Create two new columns ‘Subjectivity’ & ‘Polarity’
                            data['TextBlob_Subjectivity'] = data['clean_text'].apply(getSubjectivity)
                            data['TextBlob_Polarity'] = data['clean_text'].apply(getPolarity)

                            def getAnalysis(score):
                                if score < 0:
                                    return 'Negative'
                                elif score == 0:
                                    return 'Neutral'
                                else:
                                    return 'Positive'

                            data['TextBlob_Analysis'] = data['TextBlob_Polarity'].apply(getAnalysis)

                            # return data

                            # Layered Time Series:
                            # select date column by Time series
                            date_cols = data.select_dtypes(include=['datetime']).columns.tolist()

                            col1, col2, col3 = st.columns(3)
                            with col1:

                                #st.subheader("Select the column with date-time")
                                date_col_select = st.selectbox(label="Select the column with date-time",
                                                               options=data.columns, index=1)

                            with col2:
                                #st.subheader("Select the column with User Name User ID")
                                from_col = st.selectbox(label="Select the column with User Name User ID", options=data.columns,
                                                        index=4)

                            with col3:
                                #st.subheader("Filter by User name or ID")
                                people = data[from_col].unique()

                                person = st.selectbox(label="Filter by Username or ID", options=people, index=0)
                            st.success("There were {} messages from the selected input.".format(len(people)))

                            if date_col_select and from_col is not None:
                                                                # Convert utc to datetime
                                #data[date_col_select] = data[date_col_select].apply(lambda x: datetime.fromtimestamp(x))

                                data["datetime"] = pd.to_datetime(data[date_col_select])
                                data.index = data['datetime']
                                date_df = data.resample("D").sum()
                                date_df.reset_index(inplace=True)

                                date_df.groupby("datetime").agg(
                                    {"TextBlob_Polarity": "sum", 'TextBlob_Subjectivity': 'sum'})

                                text_df = data[col_select].dropna()
                                text = " ".join(review for review in data[col_select].dropna() if
                                                review is not None and type(review) == str)
                                st.success("There are {} words in the selected text input data.".format(len(text)))

                                col1, col2 = st.columns(2)
                                with col1:
                                    fig1 = px.line(date_df, x="datetime", y="TextBlob_Polarity",
                                                   title="Sentiment aggregate across timeline")
                                    fig1.update_xaxes(nticks=30)
                                    # sent_by_date=fig1.show()
                                    st.plotly_chart(fig1)

                                # Uncomment for subjectivity plot
                                # with col2:
                                #    fig2 = px.line(date_df, x="datetime", y="TextBlob_Subjectivity", title="Subjectivity across timeline")
                                #    fig2.update_xaxes(nticks=30)
                                #    #subj_by_date=fig2.show()
                                #    st.plotly_chart(fig2)                                    

                                # word_count across timeline

                                def get_words_count(row):
                                    message = row.clean_text
                                    emojis = ""
                                    # Telegram may save some messages as json
                                    if message is None or type(message) != str:
                                        return None
                                    return re.sub("[^\w]", " ", message).split().__len__()

                                data["word_count"] = data[["clean_text"]].apply(get_words_count, axis=1)

                                W_count = px.line(data, x="datetime", y="word_count",
                                                  title="Total length of conversation across timeline")
                                w_count_axes = W_count.update_xaxes(nticks=30)
                                # subj_by_date=fig2.show()
                                with col2:
                                    st.plotly_chart(w_count_axes)

                                    ##########################################################
                                # word count
                                ##########################################################

                                col1, col2 = st.columns(2)
                                with col1:
                                    w_count_sort = data["word_count"].resample("D").sum().sort_values(
                                        ascending=False).head(10)
                                    w_count_sort_axes = px.bar(w_count_sort,
                                                               title="When was the most conversations across timeline ? [Top 10]")
                                    fig_sort = go.Figure(w_count_sort_axes)
                                    st.plotly_chart(w_count_sort_axes)
                                    # st.pyplot(hrs_D.plot.barh(x=hrs_D.index, y=hrs_D.values, title="Top 10 words by day"))

                                data["hour"] = data.datetime.dt.hour
                                with col2:
                                    w_count_hrs = data.groupby("hour")["word_count"].sum().head(24)
                                    w_count_hrs_axes = px.bar(w_count_hrs,
                                                              title="Which hour of the day were the users active?")
                                    fig_hrs = go.Figure(w_count_hrs_axes)
                                    st.plotly_chart(fig_hrs)

                                # sumy_10=data[["datetime","hour",from_col,"word_count","clean_text"]]
                                sumy_10 = data[[from_col, "word_count", "clean_text"]]

                                sumy_10 = sumy_10.sort_values(by=["word_count"], ascending=False)
                                sumy_10 = sumy_10.head(10)
                                # st.dataframe(sumy_10)

                                sumy_filtered_data = sumy_10['clean_text']
                                sumy_topic_corpus = filtered_data.astype(str)
                                # topic_text = topic_corpus.values.tolist()

                                st.subheader('Word cloud on Top 10 by conversation length')

                                sumy_words = " "
                                stopwords = stopwords

                                # Iterate through csv file
                                for val in sumy_topic_corpus:

                                    # typecaste each val to string
                                    val = str(val)

                                    # split the vale
                                    tokens = val.split(maxsplit=2)

                                    # Converts each token into lowercase
                                    for i in range(len(tokens)):
                                        tokens[i] = tokens[i].lower()

                                    sumy_words += " ".join(tokens) + " "
                                # width=1600,height=800,
                                wordcloud = WordCloud(background_color="White",
                                                      stopwords=stopwords, min_font_size=10).generate(sumy_words)

                                # ploting the word cloud overall texts
                                sumy_fig = plt.figure(figsize=(8, 8), facecolor=None)
                                plt.imshow(wordcloud, interpolation='bilinear')
                                plt.axis("off")
                                plt.tight_layout(pad=0)
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write('**Word cloud on Top 10 by conversation length**')
                                    st.pyplot(sumy_fig)

                                    # Download file

                                towrite = io.BytesIO()
                                downloaded_file = sumy_10.to_excel(towrite, encoding='utf-8', index=False, header=True)
                                towrite.seek(0)  # reset pointer
                                b64 = base64.b64encode(towrite.read()).decode()  # some strings
                                linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Most_word_count.xlsx">Download excel with top 10 word counts</a>'
                                st.markdown(linko, unsafe_allow_html=True)

                                def dayofweek(i):
                                    l = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                    return l[i];

                                day_df = pd.DataFrame(data["word_count"])
                                day_df['day_of_date'] = data['datetime'].dt.weekday
                                day_df['day_of_date'] = day_df["day_of_date"].apply(dayofweek)
                                day_df["messagecount"] = 1
                                day = day_df.groupby("day_of_date").sum()
                                day.reset_index(inplace=True)

                                fig = px.line_polar(day, r='messagecount', theta='day_of_date', line_close=True,
                                                    title="How active were the users on each day of the week")
                                fig.update_traces(fill='toself')
                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True
                                        )),
                                    showlegend=False
                                )
                                with col2:
                                    # fig.show()
                                    st.write("How active were the users on each day of the week")
                                    st.plotly_chart(fig)

                                # Filtering by name
                                #####################
                                d = []
                                for name in people:
                                    user_df = data[data[from_col] == name]
                                    words_per_message = np.sum(user_df['word_count'])
                                    average_words = words_per_message / len(user_df)
                                    d.append(
                                        {
                                            'name': name,
                                            'length of chats': int(words_per_message),
                                            'word average per message': words_per_message / user_df.shape[0]
                                        }
                                    )

                            st.subheader('User stats')
                            user_stat_df = pd.DataFrame(d)
                            st.dataframe(user_stat_df)

                            # Download file

                            towrite = io.BytesIO()
                            downloaded_file = user_stat_df.to_excel(towrite, encoding='utf-8', index=False, header=True)
                            towrite.seek(0)  # reset pointer
                            b64 = base64.b64encode(towrite.read()).decode()  # some strings
                            linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="User_Metrics.xlsx">Download excel with user metrics</a>'
                            st.markdown(linko, unsafe_allow_html=True)

                            #########################
                            # Sentiment Analysis
                            #########################

                        elif options == "Sentiment Analysis" and submit:
                            st.subheader("""Sentiment Analysis
                            Provides an agregate of the Sentiment classification of each text message or comment.
                            
                            """)

                            data['vader_comp'] = data.apply(lambda x: sentiment_analyzer_scores(x.clean_text), axis=1,
                                                            result_type='expand')

                            data['vader_prediction'] = data.apply(lambda x: 'positive' if x.vader_comp > 0 else (
                                'negative' if x.vader_comp < 0 else 'neutral'), axis=1)

                            valence = open(proj_dir + "/vader_lexicon.txt", "r")

                            # split the vader.txt file contents with respect to spaces
                            lex_dict = {}
                            for line in valence:
                                (word, measure) = line.strip().split('\t')[0:2]
                                lex_dict[word] = float(measure)

                            # split those values that are less than '0' as negative and vice versa
                            tokenized_words_neg = dict((k, v) for k, v in lex_dict.items() if v < 0)
                            tokenized_words_pos = dict((k, v) for k, v in lex_dict.items() if v > 0)

                            data['neg_keywords'] = data.apply(
                                lambda row: [word for word in word_tokenize(row['clean_text']) if
                                             word.lower() in tokenized_words_neg], axis=1)
                            data['pos_keywords'] = data.apply(
                                lambda row: [word for word in word_tokenize(row['clean_text']) if
                                             word.lower() in tokenized_words_pos], axis=1)

                            data.to_excel(proj_dir + "/Sentiments.xlsx")

                            col1, col2, col3 = st.columns(3)

                            # Word cloud overall text
                            topic_corpus = topic_text  # .values.tolist()
                            # topic_corpus = ' '.join(map(str, topic_corpus))

                            # topic_corpus_words = list(sent_to_words(topic_corpus))

                            comment_words = " "
                            stopwords = stopwords

                            # Iterate through csv file
                            for val in topic_corpus:

                                # typecaste each val to string
                                val = str(val)

                                # split the vale
                                tokens = val.split(maxsplit=2)

                                # Converts each token into lowercase
                                for i in range(len(tokens)):
                                    tokens[i] = tokens[i].lower()

                                comment_words += " ".join(tokens) + " "
                            # width=1600,height=800,
                            wordcloud = WordCloud(background_color="White",
                                                  stopwords=stopwords, min_font_size=10).generate(comment_words)

                            # ploting the word cloud overall texts
                            fig1 = plt.figure(figsize=(8, 8), facecolor=None)
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis("off")
                            plt.tight_layout(pad=0)
                            with col1:
                                st.write('**WC on overall input**')
                                st.pyplot(fig1)

                            #
                            pos_comment_words = " "
                            # Iterate through csv file
                            for val in data.pos_keywords:

                                # typecaste each val to string
                                val = str(val)

                                # split the vale
                                tokens = val.split(maxsplit=2)

                                # Converts each token into lowercase
                                for i in range(len(tokens)):
                                    tokens[i] = tokens[i].lower()

                                pos_comment_words += " ".join(tokens) + " "
                            # width=1600,height=800,
                            wordcloud = WordCloud(background_color="White",
                                                  stopwords=stopwords, min_font_size=10).generate(pos_comment_words)

                            # ploting the word cloud overall texts
                            fig2 = plt.figure(figsize=(8, 8), facecolor=None)
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis("off")
                            plt.tight_layout(pad=0)
                            with col2:
                                st.write('**WC on Positive keywords**')
                                st.pyplot(fig2)

                            #
                            neg_comment_words = " "
                            # Iterate through csv file
                            for val in data.neg_keywords:

                                # typecaste each val to string
                                val = str(val)

                                # split the vale
                                tokens = val.split(maxsplit=2)

                                # Converts each token into lowercase
                                for i in range(len(tokens)):
                                    tokens[i] = tokens[i].lower()

                                neg_comment_words += " ".join(tokens) + " "
                            # width=1600,height=800,
                            wordcloud = WordCloud(background_color="White",
                                                  stopwords=stopwords, min_font_size=10).generate(neg_comment_words)

                            # ploting the word cloud overall texts
                            fig3 = plt.figure(figsize=(8, 8), facecolor=None)
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis("off")
                            plt.tight_layout(pad=0)
                            with col3:
                                st.write('**WC on negative keywords**')
                                st.pyplot(fig3)

                            col1, col2, col3 = st.columns(3)

                            with col2:
                                sent_count = data['vader_prediction'].value_counts()
                                # sent_count = sent_count[:4,]
                                plt.figure(figsize=(10, 5))
                                sns.barplot(sent_count.index, sent_count.values, alpha=0.8)
                                plt.title('Sentiment count across chat data')
                                plt.ylabel('Number of Occurrences', fontsize=12)
                                plt.xlabel('Sentiment polarity', fontsize=12)
                                plt.xticks(rotation=45)
                                for p in plt.gca().patches:
                                    plt.annotate("%.0f" % p.get_height(),
                                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                                 textcoords='offset points')
                                sent = plt.show()
                                st.pyplot(sent)

                            # data=pd.read_excel('/Volumes/GoogleDrive/My Drive/HAMI/Production/Application/output/Sentiments.xlsx')
                            # data=data[["clean_text","vader_prediction","neg_keywords","pos_keywords"]]
                            # st.dataframe(data)
                            # Export to excel
                            towrite = io.BytesIO()
                            downloaded_file = data.to_excel(towrite, encoding='utf-8', index=False, header=True)
                            towrite.seek(0)  # reset pointer
                            b64 = base64.b64encode(towrite.read()).decode()  # some strings
                            linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Sentiment_Prediction.xlsx">Download Sentiment_predictions file</a>'
                            st.markdown(linko, unsafe_allow_html=True)

                            #########################
                            # Emotion Analysis
                            #########################


                        elif options == "Emotion Analysis" and submit:
                            st.subheader("""Emotion Analysis
                            Identify the emotion of the text or comment and aggregates the count of the emotion in overall input text.)
                            """)

                            filename = model_dir + '/emotion_classifier_pipe_lr_Nov_2021.pkl'
                            # LogisticRegression Pipeline
                            pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression())])
                            pipeline_file = open(filename, 'rb')
                            loaded_model = joblib.load(filename)
                            data['emotion_text'] = data[col_select].apply(nfx.remove_userhandles)

                            data['emotion_text'] = data['emotion_text'].apply(nfx.remove_userhandles)

                            # User html_tags
                            data['emotion_text'] = data['emotion_text'].apply(nfx.remove_html_tags)

                            # Stopwords
                            data['emotion_text'] = data['emotion_text'].apply(nfx.remove_urls)

                            data['emotion_predict'] = data['emotion_text'].apply(
                                lambda x: (loaded_model.predict([x]))[0])

                            # plot the distribution of the predicted emotions
                            emot_count = data['emotion_predict'].value_counts()

                            col1, col2, col3 = st.columns(3)

                            with col2:

                                plt.figure(figsize=(5, 5))
                                sns.barplot(emot_count.index, emot_count.values, alpha=0.8)
                                plt.title('Emotion Analysis')
                                plt.ylabel('Number of Occurrences', fontsize=12)
                                plt.xlabel('Emotions Expressed in the text', fontsize=12)
                                plt.xticks(rotation=45)
                                # annotation on chart
                                for p in plt.gca().patches:
                                    plt.annotate("%.0f" % p.get_height(),
                                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                                 textcoords='offset points')
                                emot = plt.show()
                                st.pyplot(emot)

                            st.dataframe(data)

                            # Export to excel
                            towrite = io.BytesIO()
                            downloaded_file = data.to_excel(towrite, encoding='utf-8', index=False, header=True)
                            towrite.seek(0)  # reset pointer
                            b64 = base64.b64encode(towrite.read()).decode()  # some strings
                            linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Emotion_Prediction.xlsx">Download Emotion_predictions file</a>'
                            st.markdown(linko, unsafe_allow_html=True)

                        elif options == "Twitter Sentiment Analysis" and submit:
                            st.subheader("""Twitter Scrapper
                            Scrape the tweets from the twitter account and predict the sentiment of the tweets.
                            Search by Twitter handle or Twitter Keyword
                            """)

                            # # # # TWITTER CLIENT # # # #
                            class TwitterClient():
                                def __init__(self, twitter_user=None):
                                    self.auth = TwitterAuthenticator().authenticate_twitter_app()
                                    self.twitter_client = API(self.auth)

                                    self.twitter_user = twitter_user

                                def get_twitter_client_api(self):
                                    return self.twitter_client

                                def get_user_timeline_tweets(self, num_tweets):
                                    tweets = []
                                    for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(
                                            num_tweets):
                                        tweets.append(tweet)
                                    return tweets

                                def get_friend_list(self, num_friends):
                                    friend_list = []
                                    for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(
                                            num_friends):
                                        friend_list.append(friend)
                                    return friend_list

                                def get_home_timeline_tweets(self, num_tweets):
                                    home_timeline_tweets = []
                                    for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(
                                            num_tweets):
                                        home_timeline_tweets.append(tweet)
                                    return home_timeline_tweets

                            # # # # TWITTER AUTHENTICATER # # # #
                            class TwitterAuthenticator():

                                def authenticate_twitter_app(self):
                                    auth = OAuthHandler(twitter_credentials.CONSUMER_KEY,
                                                        twitter_credentials.CONSUMER_SECRET)
                                    auth.set_access_token(twitter_credentials.ACCESS_TOKEN,
                                                          twitter_credentials.ACCESS_TOKEN_SECRET)
                                    return auth

                            # # # # TWITTER STREAMER # # # #
                            class TwitterStreamer():
                                """
                                Class for streaming and processing live tweets.
                                """

                                def __init__(self):
                                    self.twitter_autenticator = TwitterAuthenticator()

                                def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
                                    # This handles Twitter authetification and the connection to Twitter Streaming API
                                    listener = TwitterListener(fetched_tweets_filename)
                                    auth = self.twitter_autenticator.authenticate_twitter_app()
                                    stream = Stream(auth, listener)

                                    # This line filter Twitter Streams to capture data by the keywords: 
                                    stream.filter(track=hash_tag_list)

                            # # # # TWITTER STREAM LISTENER # # # #
                            class TwitterListener(Stream):
                                """
                                This is a basic listener that just prints received tweets to stdout.
                                """

                                def __init__(self, fetched_tweets_filename):
                                    self.fetched_tweets_filename = fetched_tweets_filename

                                def on_data(self, data):
                                    try:
                                        print(data)
                                        with open(self.fetched_tweets_filename, 'a') as tf:
                                            tf.write(data)
                                        return True
                                    except BaseException as e:
                                        print("Error on_data %s" % str(e))
                                    return True

                                def on_error(self, status):
                                    if status == 420:
                                        # Returning False on_data method in case rate limit occurs.
                                        return False
                                    print(status)

                            class TweetAnalyzer():
                                """
                                Functionality for analyzing and categorizing content from tweets.
                                """

                                def clean_tweet(self, tweet):
                                    return ' '.join(
                                        re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

                                def analyze_sentiment(self, tweet):
                                    analysis = TextBlob(self.clean_tweet(tweet))

                                    if analysis.sentiment.polarity > 0:
                                        return 1
                                    elif analysis.sentiment.polarity == 0:
                                        return 0
                                    else:
                                        return -1

                                def tweets_to_data_frame(self, tweets):
                                    df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])

                                    df['id'] = np.array([tweet.id for tweet in tweets])
                                    df['len'] = np.array([len(tweet.text) for tweet in tweets])
                                    df['date'] = np.array([tweet.created_at for tweet in tweets])
                                    df['source'] = np.array([tweet.source for tweet in tweets])
                                    df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
                                    df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

                                    return df

                            if __name__ == '__main__':

                                twitter_client = TwitterClient()
                                tweet_analyzer = TweetAnalyzer()
                                twitter_streamer = TwitterStreamer()
                                
                                col1,col2=st.columns(2)
                                with col1:
                                    tweet_count = st.slider("How many tweets do you want to get?", min_value=50,
                                    max_value=10000, step=20, value=100)
                                with col2:    
                                    lang=st.multiselect('Language Selection',['en','zh','es','fr','de','it','ja','ko','pt','ru','tr','ar','bg','ca','cs','da','el','fi','he','hu','id','is','nb','nl','pl','ro','sk','sv','uk','vi','th','tl'],
                                    default='en')
                                api_hand = twitter_client.get_twitter_client_api()
                                
                                if st.checkbox('Search by twitter keywords/hashtags'):
                                    st.subheader('Analyse with results from twitter keywords/hashtags')
                                    hash_tag_list = st.text_input('Enter keyword or hash tags to search for')


                                    if hash_tag_list is not None:
                                        try:
                                            tweets = []
                                            id = []
                                            date = []
                                            source = []
                                            likes = []
                                            retweets = []

                                            for tweets_hash in tw.Cursor(api_hand.search_tweets, q=hash_tag_list,lang=lang,count=tweet_count).items(tweet_count):
                                                tweets.append(tweets_hash.text)
                                                id.append(tweets_hash.id)
                                                date.append(tweets_hash.created_at)
                                                source.append(tweets_hash.source)
                                                likes.append(tweets_hash.favorite_count)
                                                retweets.append(tweets_hash.retweet_count)
                                            df_hash = pd.DataFrame(
                                                data={'tweets': tweets, 'id': id, 'date': date, 'source': source,
                                                      'likes': likes, 'retweets': retweets})

                                            df_hash['sentiment'] = np.array(
                                                [tweet_analyzer.analyze_sentiment(tweet) for tweet in
                                                 df_hash['tweets']])

                                            st.write('Excel file with results from twitter keywords/hashtags')
                                            st.dataframe(df_hash)

                                            # plot the distribution of the predicted emotions
                                            tweet_sent_count_hash = df_hash['sentiment'].value_counts()

                                            col1, col2, col3 = st.columns(3)
                                            with col2:
                                                plt.figure(figsize=(5, 5))
                                                sns.barplot(tweet_sent_count_hash.index, tweet_sent_count_hash.values,
                                                            alpha=0.8)
                                                plt.title('Sentiment Analysis')
                                                plt.ylabel('Number of Occurrences', fontsize=12)
                                                plt.xlabel('Sentiments Expressed in the tweets', fontsize=12)
                                                plt.xticks(rotation=45)

                                                # annotation on chart
                                                for p in plt.gca().patches:
                                                    plt.annotate("%.0f" % p.get_height(),
                                                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                                                 ha='center', va='center', fontsize=10, color='black',
                                                                 xytext=(0, 5),
                                                                 textcoords='offset points')
                                                tweet_sent_plot_hash = plt.show()
                                                st.pyplot(tweet_sent_plot_hash)

                                                # Convert date-time to readable format
                                            print(df_hash['tweets'])
                                            date_columns = df_hash.select_dtypes(
                                                include=['datetime64[ns, UTC]']).columns
                                            for date_column in date_columns:
                                                df_hash[date_column] = df_hash[date_column].dt.date

                                            # Export to excel
                                            towrite = io.BytesIO()
                                            downloaded_file = df_hash.to_excel(towrite, encoding='utf-8', index=False,
                                                                               header=True)
                                            towrite.seek(0)  # reset pointer
                                            b64 = base64.b64encode(towrite.read()).decode()  # some strings
                                            linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Tweet_hashtag_results.xlsx">Download Sentiment_predictions file</a>'
                                            st.markdown(linko, unsafe_allow_html=True)

                                        except:
                                            st.warning('No tweets found')

                                if st.checkbox('Search by twitter handle names'):
                                    st.subheader('Analyse with results from twitter handle names')
                                    twitter_handle = st.text_input("Enter the Twitter Handle")

                                    tweets = []
                                    id = []
                                    date = []
                                    source = []
                                    likes = []
                                    retweets = []

                                    # Only iterate through the first 200 statuses
                                    for tweets_hand in tw.Cursor(api_hand.user_timeline,lang=lang,screen_name=twitter_handle,count=tweet_count).items(tweet_count):
                                        tweets.append(tweets_hand.text)
                                        id.append(tweets_hand.id)

                                        date.append(tweets_hand.created_at)
                                        source.append(tweets_hand.source)
                                        likes.append(tweets_hand.favorite_count)
                                        retweets.append(tweets_hand.retweet_count)

                                    df_hand = pd.DataFrame(
                                        data={'tweets': tweets, 'id': id, 'date': date, 'source': source,
                                              'likes': likes, 'retweets': retweets})

                                    df_hand['sentiment'] = np.array(
                                        [tweet_analyzer.analyze_sentiment(tweet) for tweet in df_hand['tweets']])

                                    if twitter_handle is not None:

                                        try:
                                            st.write('Excel file with desired results from tweet handle')
                                            st.dataframe(df_hand)

                                            # plot the distribution of the predicted emotions
                                            tweet_sent_count_hand = df_hand['sentiment'].value_counts()
                                            col1, col2, col3 = st.columns(3)

                                            with col2:

                                                plt.figure(figsize=(5, 5))
                                                sns.barplot(tweet_sent_count_hand.index, tweet_sent_count_hand.values,
                                                            alpha=0.8)
                                                plt.title('Sentiment Analysis')
                                                plt.ylabel('Number of Occurrences', fontsize=12)
                                                plt.xlabel('Sentiments Expressed in the tweets', fontsize=12)
                                                plt.xticks(rotation=45)
                                                # annotation on chart
                                                for p in plt.gca().patches:
                                                    plt.annotate("%.0f" % p.get_height(),
                                                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                                                 ha='center', va='center', fontsize=10, color='black',
                                                                 xytext=(0, 5),
                                                                 textcoords='offset points')
                                                tweet_sent_plot_hand = plt.show()
                                                st.pyplot(tweet_sent_plot_hand)

                                            # Convert date-time to readable format

                                            date_columns = df_hand.select_dtypes(
                                                include=['datetime64[ns, UTC]']).columns
                                            for date_column in date_columns:
                                                df_hand[date_column] = df_hand[date_column].dt.date

                                            # Export to excel

                                            towrite = io.BytesIO()
                                            downloaded_file = df_hand.to_excel(towrite, encoding='utf-8', index=False,
                                                                               header=True)
                                            towrite.seek(0)  # reset pointer
                                            b64 = base64.b64encode(towrite.read()).decode()  # some strings
                                            linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Tweet_handle_results.xlsx">Download tweet_results file</a>'
                                            st.markdown(linko, unsafe_allow_html=True)

                                        except:
                                            st.warning("Please enter a valid twitter handle")

                        ####REDDIT TEXT ANALYSIS####    
                        elif options == 'Reddit text analysis' and submit:
                            st.subheader("""Reddit text analysis
                            Reddit is a website that allows users to post text, images, videos, and links.
                            This tool allows you to analyse the sentiment of the text in the posts.
                            The search can be made by subreddit keyword or web URL.
                            """)

                            # Select the subreddit

                            if st.checkbox('Search by Subreddit name or topic'):
                                st.subheader('Analyse posts from selected Subreddit topic')
                                


                                col1,col2=st.columns(2)
                                with col1:
                                    reddit_count = st.slider("How many top posts do you want to get?", min_value=10,
                                    max_value=10000, step=10, value=15)
                                with col2:
                                    subreddit_topic = st.text_input('Provide a subreddit name to search for','spiderman')


                                #reddit = praw.Reddit(client_id='a98a0mNa_GgWIdEPYMeztQ', client_secret='YZwWwobAMtVzsPI_Zz09mAOwI48tkA', user_agent='chaitanya@AOL')
                                reddit = praw.Reddit(client_id=reddit_credentials.my_client_id, client_secret=reddit_credentials.my_client_secret, user_agent=reddit_credentials.my_user_agent)
                                posts = []
                                ml_subreddit = reddit.subreddit(subreddit_topic)
                                for post in ml_subreddit.hot(limit=reddit_count):
                                    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
                                posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

                                # Convert utc to datetime
                                posts['created'] = posts['created'].apply(lambda x: datetime.fromtimestamp(x))

                                col1,col2=st.columns(2)

                                with col1:
                                    st.subheader('Description of the retreived subreddit')
                                    st.write(ml_subreddit.description)
                                with col2:
                                    st.subheader('Number of subscribers')
                                    st.write(ml_subreddit.subscribers)
                                
                                    st.subheader("Export the Top posts from selected subreddit in excel")

                                    # Export to excel
                                    towrite = io.BytesIO()
                                    downloaded_file = posts.to_excel(towrite, encoding='utf-8', index=False,
                                                                        header=True)
                                    towrite.seek(0)  # reset pointer
                                    b64 = base64.b64encode(towrite.read()).decode()  # some strings
                                    linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Reddit_results.xlsx">Download subreddit texts scrapped as excel file</a>'
                                    st.markdown(linko, unsafe_allow_html=True)

                            elif st.checkbox('Search by URL of subreddit webpage'):
                                st.subheader('Analyse posts from selected URL of subreddit')
                                with st.spinner('Wait for it...'):

                                    #col1,col2=st.columns(2)
                                    #with col1:
                                    reddit_count = st.slider("How many top posts do you want to get?", min_value=10,
                                    max_value=10000, step=10, value=15)
                                    #with col2:
                                    submission_url = st.text_input('Provide a URL to search for','https://www.reddit.com/r/learnpython/comments/s52w65/im_trying_to_make_a_game_can_someone_explain_how/')


                                    #reddit = praw.Reddit(client_id='a98a0mNa_GgWIdEPYMeztQ', client_secret='YZwWwobAMtVzsPI_Zz09mAOwI48tkA', user_agent='chaitanya@AOL')
                                    reddit = praw.Reddit(client_id=reddit_credentials.my_client_id, client_secret=reddit_credentials.my_client_secret, user_agent=reddit_credentials.my_user_agent)
                                    commentss = []                              

                                    submission = reddit.submission(url=submission_url)
                                    submission.comments.replace_more(limit=reddit_count)
                                    for comment in submission.comments.list():
                                        commentss.append([comment.author,comment.id,comment.subreddit,comment.body,comment.created])
                                        #([comment.title, comment.score,  comment.url, comment.num_comments, comment.selftext, comment.created])
                                    commentz = pd.DataFrame(commentss,columns=['Author','ID','Subreddit','Comments','Created'])
                                    
                                    # Convert utc to datetime
                                    commentz['Created'] = commentz['Created'].apply(lambda x: datetime.fromtimestamp(x))
                                

                                    
                                    st.subheader("Export the Top posts from selected subreddit in excel")

                                # Export to excel
                                towrite = io.BytesIO()
                                downloaded_file = commentz.to_excel(towrite, encoding='utf-8', index=False,
                                                                    header=True)
                                towrite.seek(0)  # reset pointer
                                b64 = base64.b64encode(towrite.read()).decode()  # some strings
                                linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Reddit_comment_results.xlsx">Download comments scrapped as excel file</a>'
                                st.markdown(linko, unsafe_allow_html=True)                                


                #st.checkbox("Numerical Data")

            else:
                st.warning("IncorrectUsername/Password")


    elif choice == "Create Account":
        col1, col2 = st.columns(2)
        with col2:
            st_lottie(
                lottie_home3,
                speed=1,
                reverse=False,
                loop=True,
                quality="medium",
                key=None
            )
        with col1:
            st.subheader("Create new account")
            new_user = st.text_input("Username")
            new_user_email = st.text_input("Email")
            new_password = st.text_input("Password", type='password')
            new_password = st.text_input("Retype Password", type='password')

            if st.button("Create Account"):
                st.balloons()
                create_usertable()
                add_userdata(new_user, new_password)
                st.success("You have successfully created a valid account")
                st.info("Login now")

    # Bottom Headers

    st.sidebar.subheader("About App")
    st.sidebar.text("Hami Data Analysis App")
    st.sidebar.info("Analytix Online Singapore")
    st.sidebar.subheader("By")
    st.sidebar.text("© Analytix Online 2022")


if __name__ == '__main__':
    main()
