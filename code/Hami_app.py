#Hide Rerun Menu options for users
hide_menu="""
<style>
#MainMenu {
  visibility:hidden;
}
footer {
    visibility:visible;
}
footer:before{
    content:'Copyright.@.2021:Analytixlabs';
    display:block;
    position:Relative;
    color:rgba(38, 39, 48, 0.4);
    padding:5px;
    top:3px;
}
<style>
"""

#PYNGROK
from pyngrok import ngrok

from typing import Optional
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

#Importing spacy for Lemmatization
import spacy
from spacy import displacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
nlp=spacy.load("en_core_web_sm")
from collections import Counter

#sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

#import streamlit
import streamlit as st
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report
import io
import base64

#Tools required for plotting and wordcloud
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import wordcloud
from wordcloud.wordcloud import STOPWORDS


#%matplotlib inline

import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim_models
st.set_option('deprecation.showPyplotGlobalUse', False)
#pyLDAvis.enable_notebook()

#@st.cache
    #Text Processing
def text_preprocessing(text):
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text)
    #Remove new line characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#@st.cache
def show_ents(doc):
  if doc.ents():
    for ent in doc.ents:
      print(ent.text+'-'+ent.label_+'-'+str(spacy.explain(ent.label_)))

  else:
    print('No named entities found')


#############################
#TOPIC MODELLING
#############################
#@st.cache
# Define tokenization function (sentence-level) 
def sent_to_words(sentences):
    for sentence in sentences:
        yield (sent_tokenize(str(sentence)))

#@st.cache
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


df=pd.read_excel('/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/input/sample_TG.xlsx')    
#st.write(df)
df1=df.rename(columns={'messages.date':'date','messages.text':'text','messages.from':'from'})
df1= df1[['date','text','from']]
df1['text'] = df1['text'].astype(str)
df1['clean_text']=df1['text'].apply(lambda text: text_preprocessing(text))
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


#Prepare Stopwords
from nltk.corpus import stopwords
stopwords=stopwords.words('english')
stopwords.extend(['list', 'nan', 'link', 'type', 'thing', 'Haha','OK'])

vectorizer = CountVectorizer(
    analyzer="word",
    min_df=5,  
    stop_words=stopwords,
    ngram_range=(1,2)
)

corpus_vectorized = vectorizer.fit_transform(corpus_lemmatized)

search_params = {
    "n_components": [i for i in range(4,6)],
}

# Initiate the Model
lda = LatentDirichletAllocation(
    max_iter=100, #default 10
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

#pyLDAvis
panel=pyLDAvis.sklearn.prepare(
    best_lda_model, corpus_vectorized, vectorizer, mds="tsne", R=50, sort_topics=False)
pyLDAvis.save_html(panel, '/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/output/vsc_output/TG_py.html')

#Creating new df with keywords count

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

#Creating new dataframe with dominant topics and probability scores

lda_output = best_lda_model.transform(corpus_vectorized)

# column names
topicnames = ["Topic" + str(i+1) for i in range(best_lda_model.n_components)]

# index names
docnames = ["Doc" + str(i+1) for i in range(len(corpus))]

# Make the pandas dataframe
TG_topics = pd.DataFrame(
    np.round(lda_output, 2), columns=topicnames, index=docnames
)

# Get dominant topic for each document
dominant_topics = np.argmax(TG_topics.values, axis=1)
TG_topics["DOMINANT_TOPIC"] = dominant_topics+1

lemma_text = pd.DataFrame(corpus_lemmatized, columns=["Lem_Text"])

#Merging all 3 dataframes

TG_topics.reset_index(inplace=True)

full_results = df1.merge(lemma_text, left_index=True, right_index=True).merge(
    TG_topics, left_index=True, right_index=True)
full_results.drop("index", axis=1, inplace=True)

#exporting to excel
full_results.to_excel("/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/output/vsc_output/TG_pyTopics.xlsx", index=False)
keywords.to_excel("/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/output/vsc_output/TG_pykeywords.xlsx", index=False)

#importing for calling with st.dataframe()
full_results=pd.read_excel("/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/output/vsc_output/TG_pyTopics.xlsx")
keywords=pd.read_excel("/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/output/vsc_output/TG_pykeywords.xlsx")

#############################
#SENTIMENT PREDICTION
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
    #return score['neg'], score['neu'], score['pos'], score['compound']
    return score['compound']

df1['vader_comp'] = df1.apply(lambda x: sentiment_analyzer_scores(x.clean_text), axis=1, result_type='expand')


df1['vader_prediction']=df1.apply(lambda x: 'positive' if x.vader_comp > 0 else ('negative' if x.vader_comp < 0 else 'neutral'),axis=1)



valence=open("/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/input/vader_lexicon.txt","r")

#split the vader.txt file contents with respect to spaces
lex_dict={}
for line in valence:
  (word,measure)=line.strip().split('\t')[0:2]
  lex_dict[word]=float(measure)

#split those values that are less than '0' as negative and vice versa
tokenized_words_neg=dict((k,v) for k, v in lex_dict.items()if v<0)
tokenized_words_pos=dict((k,v) for k, v in lex_dict.items()if v>0)

df1['neg_keywords']=df1.apply(lambda row: [word for word in word_tokenize(row['clean_text']) if word.lower() in tokenized_words_neg], axis=1)
df1['pos_keywords']=df1.apply(lambda row: [word for word in word_tokenize(row['clean_text']) if word.lower() in tokenized_words_pos], axis=1)

df1.to_excel("/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/output/Tg_sentiments.xlsx")



####################################################
#APPLYING the NER FROM SPACY
####################################################

df1['NER']=df1['clean_text'].apply(lambda x: nlp(x))

tokens=nlp(''.join(str(df1.clean_text.tolist())))
items=[x.text for x in tokens.ents]
#Counter(items).most_common(20)

#Names of persons
person_list = []
for ent in tokens.ents:
    if ent.label_ == 'PERSON':
        person_list.append(ent.text)
        
person_counts = Counter(person_list).most_common(20)
df_person = pd.DataFrame(person_counts, columns =['text', 'count'])




#Nationalities, religious and political groups
norp_list = []
for ent in tokens.ents:
    if ent.label_ == 'NORP':
        norp_list.append(ent.text)
        
norp_counts = Counter(norp_list).most_common(20)
df_norp = pd.DataFrame(norp_counts, columns =['text', 'count'])



#Companies Agencies institutions

org_list = []
for ent in tokens.ents:
    if ent.label_ == 'ORG':
        org_list.append(ent.text)
        
org_counts = Counter(org_list).most_common(20)
df_org = pd.DataFrame(org_counts, columns =['text', 'count'])




#Objects,vehicles,foods(not services)

prod_list = []
for ent in tokens.ents:
    if ent.label_ == 'PRODUCT':
        prod_list.append(ent.text)
        
prod_counts = Counter(prod_list).most_common(20)
df_prod = pd.DataFrame(prod_counts, columns =['text', 'count'])



#Countres, cities and states
gpe_list = []
for ent in tokens.ents:
    if ent.label_ == 'GPE':
        gpe_list.append(ent.text)
        
gpe_counts = Counter(gpe_list).most_common(20)
df_gpe = pd.DataFrame(gpe_counts, columns =['text', 'count'])

######################
#SUMMARIZATION
######################

# Sumy Summary Package
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


#Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

#summary=df1['clean_text'].apply(lambda x: sumy_summarizer(x))
sumy=sumy_summarizer(corpus)

#sql database management to store passwords
import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password)==hashed_text:
        return hashed_text
    return False    

import sqlite3
conn=sqlite3.connect('data.db')
c=conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS usertable(username TEXT,password TEXT,email TEXT)')

def add_userdata(username,password):
    c.execute('INSERT INTO usertable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * from usertable WHERE username = ? AND password = ?',(username,password))
    data=c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM usertable')
    data=c.fetchall()
    return data

#select columns
def select_cols(self):
    if self.df is not None:
        try:
            df_columns=list(self.df.columns)
            label='Select Columns'
            self.usecols=st.multiselect(label=label, default=df_columns,options=df_columns)
            self.df=self.df[self.usecols]
        except:
            st.write('Column Select Error')
            


def main():
    #Top Headers
    st.sidebar.image('/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/codes/HAMI LOGO.png')
    st.sidebar.title("ANALYTIX LABS")
    st.sidebar.subheader("AI Solutions, Fine-Tuned to You")

    #Hide Menu and edit Footer
    st.markdown(hide_menu,unsafe_allow_html=True)

    #streamlit commands
    st.title("HAMI Data Analytics App")
    menu=["Home","Login","Create Account"]
    choice=st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home")
    elif choice == "Login":
        st.subheader("Login Section")

        username=st.sidebar.text_input("User Name")
        #email=st.sidebar.text_input("Email")
        password=st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            # if password=='admin':
            create_usertable()
            result=login_user(username,password)
            if result:    
                st.success("Logged In as {}".format(username))

                #Header/Subheader
                st.header("Analysis")

                #Uploader
                
                dff=st.file_uploader("Choose a file to Upload",
                    type=["xlsx","csv"])
                if dff is not None:
                    def load_csv():
                        csv=pd.read_csv(dff)
                        return csv
                    
                    #File details
                    file_details = {"Filename":dff.name,
                    "FileType":dff.type,"FileSize":dff.size}
                    #read excel
                    data = pd.read_excel(dff)
                    #read csv
                    #data= load_csv()
                    pr=ProfileReport(data,explorative=True)
                    st.dataframe(data)
                    st.header("**Pandas profiling report**")
                    st_profile_report(pr)
                else:
                    st.warning("you need to upload a csv or excel file to start the analysis")






                
                if st.checkbox("Textual Data"):

                #SelectBox
                    options=st.radio("Select the task",["Topic Modelling","Entity analysis","Sentiment Prediction","Text Summarization"])
                    st.write("Select the Entity to see results")
                    if options == "Entity analysis":
                        select_plot=st.radio("Entity types identified",("Person : People including fictional"
                        ,"NORP : Nationalities or religious or political groups"
                        ,"ORG : Companies, agencies, institutions."
                        ,"GPE : Countries, cities, states"
                        ,"PRODUCT : Objects, vehicles, foods"))
                        if select_plot == "Person : People including fictional":
                            st.pyplot(df_person.plot.barh(x='text', y='count', title="Names of persons", color="#80852c", figsize=(10,8)).invert_yaxis())
                        elif select_plot == "NORP : Nationalities or religious or political groups":
                            st.pyplot(df_norp.plot.barh(x='text', y='count',color="#e0c295", title="Nationalities, religious and political groups", figsize=(10,8)).invert_yaxis())
                        elif select_plot == "ORG : Companies, agencies, institutions.":
                            st.pyplot(df_org.plot.barh(x='text', y='count',color="#cdbb69",title="Companies Agencies institutions", figsize=(10,8)).invert_yaxis())
                        elif select_plot == "PRODUCT : Objects, vehicles, foods":
                            st.pyplot(df_prod.plot.barh(x='text', y='count',color="#eca349",title="Objects,vehicles,foods", figsize=(10,8)).invert_yaxis())
                        else:
                            st.pyplot(df_gpe.plot.barh(x='text', y='count',color="#e08a31" ,title="Countres, cities and states", figsize=(10,8)).invert_yaxis())
                    elif options == "Topic Modelling":
                        components.html(
                            """
                        <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v1.0.0.css">


                        <div id="ldavis_el310621402521896492648484151214"></div>
                        <script type="text/javascript">

                        var ldavis_el310621402521896492648484151214_data = {"mdsDat": {"x": [-60.31231689453125, 60.31256103515625], "y": [141.36962890625, -141.369140625], "topics": [1, 2], "cluster": [1, 1], "Freq": [51.99171190517975, 48.00828809482025]}, "tinfo": {"Term": ["vaccine", "covid", "say", "group", "adverse", "still", "thank", "report", "injury", "even", "case", "know", "one", "19", "vaccines", "really", "people", "reach", "join", "great", "channel", "hear", "want", "vax", "time", "see", "also", "long", "share", "group", "know", "one", "case", "really", "people", "reach", "join", "great", "channel", "hear", "want", "vax", "time", "see", "also", "long", "share", "even", "vaccines", "19", "injury", "report", "thank", "still", "adverse", "say", "covid", "vaccine", "vaccine", "covid", "say", "adverse", "still", "thank", "report", "injury", "19", "vaccines", "even", "share", "long", "also", "see", "time", "vax", "want", "hear", "channel", "great", "join", "reach", "people", "really", "case", "one", "know", "group"], "Freq": [15.0, 8.0, 8.0, 8.0, 7.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 6.0, 6.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 7.884872972726452, 6.211247100767315, 6.185249045533402, 6.936301676052613, 6.1621153972599, 5.390433486961799, 4.564393009304724, 4.564281867552449, 4.561181097772296, 4.560068201492858, 4.547045217605094, 4.522928549446765, 4.457023693931555, 4.40244657600597, 4.01419196952716, 5.062903180588815, 3.6432830284628035, 3.6067676962067696, 0.4996294788067733, 0.42067889392657726, 0.420199139076834, 0.4720061232202885, 0.43570547835305945, 0.43046026625063377, 0.4261574702520073, 0.4859430736134029, 0.4660014780298373, 0.4243107316019786, 0.42684112425240667, 14.897547677824795, 7.671331748777769, 7.626010963579068, 6.700712210514206, 5.862082552331848, 5.857405109255958, 5.851703193330157, 5.812241829129142, 4.964938848477392, 4.9644173210796785, 5.7822132910896835, 1.500913819602123, 1.461219075050158, 1.7252333898086087, 1.0580148658057884, 0.6359546934585066, 0.5766255125202592, 0.5049822809574973, 0.47876575976368907, 0.4646088567098044, 0.46339905987116886, 0.46002830417121915, 0.4599074852386496, 0.46556398892262896, 0.5303120936468402, 0.5923377729810715, 0.5051641844800073, 0.476902462688129, 0.46479462435162294], "Total": [15.0, 8.0, 8.0, 8.0, 7.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 6.0, 6.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 8.349667597078074, 6.688149563455443, 6.690413230013409, 7.528639449033685, 6.69242749090674, 5.855997475884427, 5.024300494543374, 5.024310171723668, 5.024580157643466, 5.024677058202663, 5.025810977368783, 5.027910830404262, 5.033649206451814, 5.038401269464477, 5.072206835332949, 6.788136570397424, 5.104502103512962, 5.107681515808893, 6.281842769896457, 5.385096215006255, 5.385137987554226, 6.28424795234943, 6.287408671683217, 6.287865375506592, 6.288240022583855, 7.1866552841276095, 8.092012441608905, 8.095642480379748, 15.324388802077202, 15.324388802077202, 8.095642480379748, 8.092012441608905, 7.1866552841276095, 6.288240022583855, 6.287865375506592, 6.287408671683217, 6.28424795234943, 5.385137987554226, 5.385096215006255, 6.281842769896457, 5.107681515808893, 5.104502103512962, 6.788136570397424, 5.072206835332949, 5.038401269464477, 5.033649206451814, 5.027910830404262, 5.025810977368783, 5.024677058202663, 5.024580157643466, 5.024310171723668, 5.024300494543374, 5.855997475884427, 6.69242749090674, 7.528639449033685, 6.690413230013409, 6.688149563455443, 8.349667597078074], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2"], "logprob": [29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -2.5013, -2.7399, -2.7441, -2.6295, -2.7478, -2.8816, -3.048, -3.048, -3.0487, -3.0489, -3.0518, -3.0571, -3.0718, -3.0841, -3.1764, -2.9443, -3.2734, -3.2835, -5.2602, -5.4322, -5.4333, -5.317, -5.3971, -5.4092, -5.4192, -5.2879, -5.3298, -5.4236, -5.4176, -1.7854, -2.4491, -2.455, -2.5843, -2.7181, -2.7189, -2.7198, -2.7266, -2.8842, -2.8843, -2.7318, -4.0805, -4.1073, -3.9412, -4.4302, -4.9392, -5.0371, -5.1698, -5.2231, -5.2531, -5.2557, -5.263, -5.2633, -5.2511, -5.1208, -5.0102, -5.1694, -5.227, -5.2527], "loglift": [29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5968, 0.5801, 0.5756, 0.5721, 0.5715, 0.5712, 0.5581, 0.5581, 0.5573, 0.5571, 0.554, 0.5482, 0.5324, 0.5192, 0.4201, 0.3608, 0.3168, 0.3062, -1.8775, -1.8954, -1.8966, -1.9347, -2.0153, -2.0274, -2.0375, -2.0398, -2.2004, -2.2945, -2.9267, 0.7055, 0.68, 0.6745, 0.6638, 0.6636, 0.6629, 0.662, 0.6557, 0.6526, 0.6525, 0.6509, -0.4909, -0.5171, -0.636, -0.8336, -1.3359, -1.4329, -1.5644, -1.6173, -1.6471, -1.6497, -1.657, -1.6572, -1.7982, -1.8015, -1.8086, -1.8498, -1.907, -2.1546]}, "token.table": {"Topic": [2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2], "Freq": [0.9284813149738537, 0.9740275167309256, 0.736579169871838, 0.2946316679487352, 0.9297828707813154, 0.13282612439733077, 0.9950888270197628, 0.9881859308125892, 0.9551337433583836, 0.9951080176109691, 0.9581219739573299, 0.9948643159312974, 0.9547681831613342, 0.9951614906538845, 0.8971091245902237, 0.7836219711315557, 0.19590549278288893, 0.8968055923786302, 0.1494675987297717, 0.8538255046369967, 0.9951634074096951, 0.896535675306521, 0.14942261255108685, 0.9542882152741833, 0.988629226379364, 0.7886113736798024, 0.1971528434199506, 0.7831341847802209, 0.39156709239011045, 0.9541620514565828, 0.9542189028683841, 0.7939026262640556, 0.1984756565660139, 0.9788318603588789, 0.9284885172649031, 0.7946521173690555, 0.19866302934226387, 0.9944488215193709, 0.19888976430387417], "Term": ["19", "adverse", "also", "also", "case", "case", "channel", "covid", "even", "great", "group", "hear", "injury", "join", "know", "long", "long", "one", "one", "people", "reach", "really", "really", "report", "say", "see", "see", "share", "share", "still", "thank", "time", "time", "vaccine", "vaccines", "vax", "vax", "want", "want"]}, "R": 29, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [1, 2]};

                        function LDAvis_load_lib(url, callback){
                        var s = document.createElement('script');
                        s.src = url;
                        s.async = true;
                        s.onreadystatechange = s.onload = callback;
                        s.onerror = function(){console.warn("failed to load library " + url);};
                        document.getElementsByTagName("head")[0].appendChild(s);
                        }

                        if(typeof(LDAvis) !== "undefined"){
                        // already loaded: just create the visualization
                        !function(LDAvis){
                            new LDAvis("#" + "ldavis_el310621402521896492648484151214", ldavis_el310621402521896492648484151214_data);
                        }(LDAvis);
                        }else if(typeof define === "function" && define.amd){
                        // require.js is available: use it to load d3/LDAvis
                        require.config({paths: {d3: "https://d3js.org/d3.v5"}});
                        require(["d3"], function(d3){
                            window.d3 = d3;
                            LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                                new LDAvis("#" + "ldavis_el310621402521896492648484151214", ldavis_el310621402521896492648484151214_data);
                            });
                            });
                        }else{
                            // require.js not available: dynamically load d3 & LDAvis
                            LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
                                LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                                        new LDAvis("#" + "ldavis_el310621402521896492648484151214", ldavis_el310621402521896492648484151214_data);
                                    })
                                });
                        }
                        </script>
                        """,height=600,width=1800,scrolling=True)
                        if st.checkbox("Dominant Topic Output"):
                            fr=pd.read_csv('/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/output/vsc_output/csv files/TG_pyTopics.csv')
                            fr=fr[["date","from","Lem_Text","DOMINANT_TOPIC"]]
                            st.dataframe(fr)
                            #Export fr to excel
                            towrite = io.BytesIO()
                            downloaded_file = fr.to_excel(towrite, encoding='utf-8', index=False, header=True)
                            towrite.seek(0)  # reset pointer
                            b64 = base64.b64encode(towrite.read()).decode()  # some strings
                            linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Dominant_Topic.xlsx">Download Dominant_Topic file</a>'
                            st.markdown(linko, unsafe_allow_html=True)
                        if st.checkbox("Keyword List"):
                            kw=pd.read_csv('/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/output/vsc_output/csv files/TG_pykeywords.csv')
                            st.dataframe(kw)
                            #Export kw to excel
                            towrite = io.BytesIO()
                            downloaded_file = kw.to_excel(towrite, encoding='utf-8', index=False, header=True)
                            towrite.seek(0)  # reset pointer
                            b64 = base64.b64encode(towrite.read()).decode()  # some strings
                            linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Keyword_list.xlsx">Download Keyword_list file</a>'
                            st.markdown(linko, unsafe_allow_html=True)
                    elif options == "Text Summarization":
                        st.success(sumy)
                    else:

                        #Word cloud
                        st.subheader('Word cloud on chat keywords')

                        comment_words=''
                        stopwords=set(STOPWORDS)

                        #Iterate through csv file
                        for val in keywords.Keyword:

                            #typecaste each val to string
                            val=str(val)

                            #split the vale
                            tokens=val.split(maxsplit=2)

                            #Converts each token into lowercase
                            for i in range(len(tokens)):
                                tokens[i]=tokens[i].lower()

                            comment_words+=" ".join(tokens)+" "

                        wordcloud=WordCloud(width=1600,height=800, background_color="White",
                        stopwords=stopwords,min_font_size=10).generate(comment_words)

                        #ploting the word cloud
                        fig=plt.figure(figsize=(8,8),facecolor=None)
                        plt.imshow(wordcloud,interpolation='bilinear')
                        plt.axis("off")
                        plt.tight_layout(pad=0)
                        st.pyplot(fig)

                        sent_count = df1['vader_prediction'].value_counts()
                        #sent_count = sent_count[:4,]
                        plt.figure(figsize=(10,5))
                        sns.barplot(sent_count.index, sent_count.values, alpha=0.8)
                        plt.title('Sentiment count across chat data')
                        plt.ylabel('Number of Occurrences', fontsize=12)
                        plt.xlabel('Sentiment', fontsize=12)
                        sent=plt.show()
                        st.pyplot(sent)

                        df2=pd.read_csv('/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/output/vsc_output/csv files/Tg_sentiments.csv')
                        df2=df2[["date","from","clean_text","vader_prediction","neg_keywords","pos_keywords"]]
                        st.dataframe(df2)
                        #Export fr to excel
                        towrite = io.BytesIO()
                        downloaded_file = df2.to_excel(towrite, encoding='utf-8', index=False, header=True)
                        towrite.seek(0)  # reset pointer
                        b64 = base64.b64encode(towrite.read()).decode()  # some strings
                        linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Sentiment_Prediction.xlsx">Download Sentiment_predictions file</a>'
                        st.markdown(linko, unsafe_allow_html=True)

                        #user_result=
                
                st.checkbox("Numerical Data")

            else:
                st.warning("IncorrectUsername/Password")


    elif choice=="Create Account":
        st.subheader("Create new account")
        new_user=st.text_input("Username")
        new_user_email=st.text_input("Email")
        new_password=st.text_input("Password",type='password')
        new_password=st.text_input("Retype Password",type='password')
        
        if st.button("Create Account"):
            create_usertable()
            add_userdata(new_user,new_password)
            st.success("You have successfully created a valid account")
            st.info("Login now")





    
    # Bottom Headers

    st.sidebar.subheader("About App")
    st.sidebar.text("Hami Data Analysis App")
    st.sidebar.info("Analytix-Labs Singapore")
    st.sidebar.subheader("By")
    st.sidebar.text("© Analytix-Labs 2021")
        




if __name__== '__main__':
    main()