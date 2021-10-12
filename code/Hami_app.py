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
    content:'Copyright.@.2021:AnalytixOnline';
    display:block;
    position:Relative;
    color:rgba(38, 39, 48, 0.4);
    padding:5px;
    top:3px;
}
<style>
"""

#PYNGROK
from os import name
from pickle import STOP
from pyngrok import ngrok

#streamlit-tags
from streamlit_tags import st_tags #pip install streamlit_tags==1.0.6

#lottie-Animation
from streamlit_lottie import st_lottie
import json
from pandas.io.json import json_normalize
import requests

from typing import Optional
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import nltk
import re

from streamlit.uploaded_file_manager import UploadedFile
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from transformers import pipeline

#Importing spacy for Lemmatization
import spacy
from spacy import displacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
nlp=spacy.load("en_core_web_sm")
nlp.max_length = 1450000 # or even higher
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
st.set_page_config(layout="wide")
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

#################################
#Lottie Animation
#################################
# Load the animation from local folder
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

#Load animation directly from website
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code!=200:
        return None
    return r.json

lottie_home=load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_3ezlslmp.json")

lottie_home2=load_lottiefile("/Volumes/GoogleDrive/My Drive/HAMI/Production/Application/code/Hami_home.json")



#%matplotlib inline

import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim_models
st.set_option('deprecation.showPyplotGlobalUse', False)
#pyLDAvis.enable_notebook()

#@st.cache
#Text Preprocessing
def text_preprocessing(text):
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text)
    #Remove emails
    text = re.sub(r'(\W|^)[\w.\-]{0,25}@(yahoo|hotmail|gmail)\.com(\W|$)', ' ', text)
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

#word cloud
#Define tokenization function (word-level) 
def sent_to_words(sentence):
    for word in sentence:
        yield (word_tokenize(str(word)))


#Prepare Stopwords

stopwords=stopwords.words('english')
stopwords.extend(['list', 'nan', 'link', 'type', 'thing', 'Haha','OK',"'lol'",'nil',"nil'","https","www","think","like","text","lol","no'","like'","text","com","2021","covid","19","vaccine","'"])



#############################
#TOPIC MODELLING
#############################
#@st.cache
# Define tokenization function (sentence-level) 
def para_to_sents(paragraph):
    for sentence in paragraph:
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

#T5 Summarization model
#model.load_model("t5","/content/outputs/simplet5-epoch-4-train-loss-0.6352", use_gpu=True)

############################################
#sql database management to store passwords
#############################################
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
    st.sidebar.title("ANALYTIX ONLINE")
    st.sidebar.subheader("AI Solutions, Fine-Tuned to You")

    #Hide Menu and edit Footer
    st.markdown(hide_menu,unsafe_allow_html=True)

    #streamlit commands
    st.title("HAMI Data Analysis App")
    menu=["Home","Login","Create Account"]
    choice=st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home")
        st_lottie(
            lottie_home2,
            speed=1,
            reverse=False,
            loop=True,
            quality="medium",
            key=None
        )

    elif choice == "Login":
        st.subheader("Login Section")

        username=st.sidebar.text_input("User Name")
        #email=st.sidebar.text_input("Email")
        password=st.sidebar.text_input("Password",type='password')
        global stopwords
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
                    type=["xlsx","csv","json"])

                global data #To make it available outside the if statement
                if dff is not None:
                    try:
                        data=pd.read_csv(dff)
                    except Exception as e:
                        data=pd.read_excel(dff)
                    #except Exception as e:
                    #    with open(dff, 'w') as f:
                    #        json.dump(dff, f)
                    #    #yy=json_normalize(dff)
                    #    data=pd.read_json(dff)
                    #File details
                    file_details = {"Filename":dff.name,
                    "FileType":dff.type,"FileSize":dff.size}
                    st.write(file_details)

                global numeric_columns
                global non_numeric_columns
                global column_options

                try:

                    numeric_columns=list(data.select_dtypes(['float','int']).columns)
                    non_numeric_columns=list(data.select_dtypes(['object']).columns)
                    #text_columns=data.select_dtypes(['object']).columns
                    column_options=list(set(non_numeric_columns))

                

                    non_numeric_columns.append(None)
                    col_select=st.selectbox(
                    label="Select the column for analysis",
                    options=data.columns
                    )
                    st.dataframe(data)
                    data[col_select] = data[col_select].astype(str)
                    data['clean_text']=data[col_select].apply(lambda text: text_preprocessing(text))

                    #DATA_COLUMN
                    data[col_select] = data[col_select].astype(str)
                    data['clean_text']=data[col_select].apply(lambda text: text_preprocessing(text))      
                    filtered_data = data[col_select]
                    topic_text = filtered_data.astype(str)
                    topic_corpus = topic_text.values.tolist()
                    topic_corpus_words = list(para_to_sents(topic_corpus))

                    






                    st.header("**Data Exploration**")
                    pr=ProfileReport(data,explorative=False,orange_mode=True,minimal=False,samples=None,correlations=None,missing_diagrams=None,duplicates=None,interactions=None,)
                    #st_profile_report(pr)




                except Exception as e:
                    print(e)
                    st.warning("you need to upload a csv or excel file to start the analysis")




                
                if st.checkbox("Textual Data"):



                    
                        ################################################
                        #Streamlit Deck
                        ################################################
                    #Submit button
                    with st.form(key="form1"):
                        #SelectBox
                        options=st.radio("Select the task",["Topic Modelling","Entity analysis","Sentiment Prediction","Text Summarization"])
                        submit=st.form_submit_button(label="Submit")

                        
                        
                        
                        
                        #########################
                        #TOPIC MODELLING
                        #########################
                        col1,col2,col3,col4=st.columns(4)   
                        with col1:
                            minimum_df=st.slider('slide to set min_df',min_value=1,max_value=8,help="Minimum required occurences of a word")
                        with col2:
                            collect_numbers = lambda x : [int(i) for i in re.split("[^0-9]", x) if i != ""]
                            number_of_topics=st.text_input('Enter number of topics(minimum is 2)')
                            ticks=(collect_numbers(number_of_topics))
                        with col3:
                            n_grams=st.slider('select a range of n-grams',1,5,(1,2),help="Assign the number ngram for keywords/phrases i.e.Bi-gram, tri-gram,...n-gram")
                        with col4:
                            keywords=st_tags('Enter custom Stopwords:','Press enter to add more',['hello'])
                            
                            stopwords.extend(keywords)

                                

                        if options == "Topic Modelling" and submit:

                        #Assignments



                                



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
                                max_iter=100, #default 10
                                learning_method="batch",
                                batch_size=32,
                                learning_decay=0.7,
                                random_state=42,)

                            # Init Grid Search Class
                            topic_model = GridSearchCV(topic_lda, cv=5, param_grid=search_params, n_jobs=-1, verbose=1)

                            # Do the Grid Search
                            topic_model.fit(topic_corpus_vectorized)

                            # LDA Model
                            topic_best_lda_model = topic_model.best_estimator_

                            #pyLDAvis
                            topic_panel=pyLDAvis.sklearn.prepare(
                                topic_best_lda_model, topic_corpus_vectorized, topic_vectorizer, mds="tsne", R=50, sort_topics=False)
                            pyLDAvis.save_html(topic_panel, '/Volumes/GoogleDrive/My Drive/HAMI/Production/Application/output/topic_panel.html')
                            
                            #Creating new df with keywords count

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


                            #Creating new dataframe with dominant topics and probability scores

                            topic_lda_output = topic_best_lda_model.transform(topic_corpus_vectorized)

                            # column names
                            Topicnames = ["Topic" + str(i+1) for i in range(topic_best_lda_model.n_components)]

                            # index names
                            Docnames = ["Doc" + str(i+1) for i in range(len(topic_corpus))]

                            # Make the pandas dataframe
                            Topics = pd.DataFrame(
                                np.round(topic_lda_output, 2), columns=Topicnames, index=Docnames
                            )

                            # Get dominant topic for each document
                            Dominant_topics = np.argmax(Topics.values, axis=1)
                            Topics["DOMINANT_TOPIC"] = Dominant_topics+1

                            topic_lemma_text = pd.DataFrame(topic_corpus_lemmatized, columns=["Lem_Text"])

                            #Merging all 3 dataframes

                            Topics.reset_index(inplace=True)

                            topic_full_results = data.merge(topic_lemma_text, left_index=True, right_index=True).merge(
                                Topics, left_index=True, right_index=True)
                            topic_full_results.drop("index", axis=1, inplace=True)

                            #exporting to excel
                            topic_full_results.to_excel("/Volumes/GoogleDrive/My Drive/HAMI/Production/Application/output/Topics.xlsx", index=False)
                            topic_keywords.to_excel("/Volumes/GoogleDrive/My Drive/HAMI/Production/Application/output/Keywords.xlsx", index=False)

                            #importing for calling with st.dataframe()
                            topic_full_results=pd.read_excel("/Volumes/GoogleDrive/My Drive/HAMI/Production/Application/output/Topics.xlsx")
                            topic_keywords=pd.read_excel("/Volumes/GoogleDrive/My Drive/HAMI/Production/Application/output/Keywords.xlsx")


                            #st.plotly_chart(topic_panel)
                            html_string = pyLDAvis.prepared_data_to_html(topic_panel)
                            from streamlit import components
                            components.v1.html(html_string, width=1300, height=900, scrolling=False)
                            
                            col1,col2=st.columns(2)
                            
                            with col1:                                                
                                st.write("**Dominant Topic Output**")
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
                                
                            with col2:
                                st.write("**Keyword List**")
                                kw=pd.read_csv('/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/output/vsc_output/csv files/TG_pykeywords.csv')
                                st.dataframe(kw)
                                #Export kw to excel
                                towrite = io.BytesIO()
                                downloaded_file = kw.to_excel(towrite, encoding='utf-8', index=False, header=True)
                                towrite.seek(0)  # reset pointer
                                b64 = base64.b64encode(towrite.read()).decode()  # some strings
                                linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Keyword_list.xlsx">Download Keyword_list file</a>'
                                st.markdown(linko, unsafe_allow_html=True)













                        elif options == "Entity analysis":




                            ####################################################
                            #ENTITY RECOGNITION : APPLYING the NER FROM SPACY
                            ####################################################
                            st.write("**Select the Entity from dropdown to see results**")

                            #data['NER']=data['clean_text'].apply(lambda x: nlp(x))

                            tokens=nlp(''.join(str(data.clean_text.tolist())))
                            items=[x.text for x in tokens.ents]
                            #Counter(items).most_common(20)


                            select_plot=st.selectbox("Entity types identified",("Person : People including fictional"
                            ,"NORP : Nationalities or religious or political groups"
                            ,"ORG : Companies, agencies, institutions."
                            ,"GPE : Countries, cities, states"
                            ,"PRODUCT : Objects, vehicles, foods"
                            ,"EVENT : Named hurricanes, battles, wars, sports events"
                            ,"LAW : Named documents made into laws"
                            ,"WORK_OF_ART: Titles of books, songs"
                            ))

                            if select_plot == "Person : People including fictional":
                                
                                #Names of persons

                                person_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'PERSON':
                                        person_list.append(ent.text)
                                person_counts = Counter(person_list).most_common(20)
                                df_person = pd.DataFrame(person_counts, columns =['text', 'count'])
                            
                                st.pyplot(df_person.plot.barh(x='text', y='count', title="Names of persons", color="#80852c", figsize=(10,8)).invert_yaxis())

                            elif select_plot == "NORP : Nationalities or religious or political groups":
                                
                                #Nationalities, religious and political groups

                                norp_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'NORP':
                                        norp_list.append(ent.text)
                                norp_counts = Counter(norp_list).most_common(20)
                                df_norp = pd.DataFrame(norp_counts, columns =['text', 'count'])                                
                            
                                st.pyplot(df_norp.plot.barh(x='text', y='count',color="#e0c295", title="Nationalities, religious and political groups", figsize=(10,8)).invert_yaxis())
                            
                            elif select_plot == "ORG : Companies, agencies, institutions.":

                                #Companies Agencies institutions

                                org_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'ORG':
                                        org_list.append(ent.text)
                                        
                                org_counts = Counter(org_list).most_common(20)
                                df_org = pd.DataFrame(org_counts, columns =['text', 'count'])

                                st.pyplot(df_org.plot.barh(x='text', y='count',color="#cdbb69",title="Companies Agencies institutions", figsize=(5,4)).invert_yaxis())

                            elif select_plot == "PRODUCT : Objects, vehicles, foods":

                                #Objects,vehicles,foods(not services)

                                prod_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'PRODUCT':
                                        prod_list.append(ent.text)
                                        
                                prod_counts = Counter(prod_list).most_common(20)
                                df_prod = pd.DataFrame(prod_counts, columns =['text', 'count'])

                                st.pyplot(df_prod.plot.barh(x='text', y='count',color="#eca349",title="Objects,vehicles,foods", figsize=(5,4)).invert_yaxis())

                            elif select_plot == "EVENT : Named hurricanes, battles, wars, sports events":

                                #EVENT: Named hurricanes, battles, wars, sports events,
                                event_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'EVENT':
                                        event_list.append(ent.text)

                                event_counts = Counter(event_list).most_common(20)
                                df_event = pd.DataFrame(event_counts, columns=['text','count'])

                                st.pyplot(df_event.plot.barh(x='text',y='count',color="#e08a31", title="Named hurricanes, battles, wars, sports events",figsize=(5,4)).invert_yaxis())
                            
                            elif select_plot == "LAW : Named documents made into laws":

                                #LAW: Named documents made into laws
                                law_list= []
                                for ent in tokens.ents:
                                    if ent.label_ == 'LAW':
                                        law_list.append(ent.text)
                                
                                law_counts = Counter(law_list).most_common(20)
                                df_law = pd.DataFrame(law_counts,columns= ['text','count'])

                                st.pyplot(df_law.plot.barh(x='text',y='count',color="#80852c", title="Named documents made into laws",figsize=(5,4)).invert_yaxis())

                            elif select_plot == "WORK_OF_ART: Titles of books, songs":
                                
                                #WORK_OF_ART: Titles of books, songs
                                workofart_list= []
                                for ent in tokens.ents:
                                    if ent.label_ == 'WORK_OF_ART':
                                        workofart_list.append(ent.text)
                                
                                workofart_counts = Counter(workofart_list).most_common(20)
                                df_workofart = pd.DataFrame(workofart_counts,columns= ['text','count'])

                                st.pyplot(df_workofart.plot.barh(x='text',y='count',color="#e0c295", title="Named documents made into laws",figsize=(5,4)).invert_yaxis())


                            
                            else:

                                #Countres, cities and states

                                gpe_list = []
                                for ent in tokens.ents:
                                    if ent.label_ == 'GPE':
                                        gpe_list.append(ent.text)
                                        
                                gpe_counts = Counter(gpe_list).most_common(20)
                                df_gpe = pd.DataFrame(gpe_counts, columns =['text', 'count'])

                                st.pyplot(df_gpe.plot.barh(x='text', y='count',color="#e08a31" ,title="Countres, cities and states", figsize=(10,8)).invert_yaxis()) 

                            
                                 
                                                       
                                                        
                            
                            #########################
                            #Text Summarization
                            #########################    
                            
                        elif options == "Text Summarization":

                            summary=data['clean_text'].apply(lambda x: sumy_summarizer(x))
                            sumy=sumy_summarizer(topic_corpus)
                            st.success(sumy)

                            #########################
                            #Sentiment Analysis
                            #########################

                        elif options == "Sentiment Prediction":
                            
                            data['vader_comp'] = data.apply(lambda x: sentiment_analyzer_scores(x.clean_text), axis=1, result_type='expand')

                            data['vader_prediction']=data.apply(lambda x: 'positive' if x.vader_comp > 0 else ('negative' if x.vader_comp < 0 else 'neutral'),axis=1)

                            valence=open("/Volumes/GoogleDrive/My Drive/HAMI/Partners/HAN EI-TG/input/vader_lexicon.txt","r")

                            #split the vader.txt file contents with respect to spaces
                            lex_dict={}
                            for line in valence:
                                (word,measure)=line.strip().split('\t')[0:2]
                                lex_dict[word]=float(measure)

                            #split those values that are less than '0' as negative and vice versa
                            tokenized_words_neg=dict((k,v) for k, v in lex_dict.items()if v<0)
                            tokenized_words_pos=dict((k,v) for k, v in lex_dict.items()if v>0)

                            data['neg_keywords']=data.apply(lambda row: [word for word in word_tokenize(row['clean_text']) if word.lower() in tokenized_words_neg], axis=1)
                            data['pos_keywords']=data.apply(lambda row: [word for word in word_tokenize(row['clean_text']) if word.lower() in tokenized_words_pos], axis=1)

                            data.to_excel("/Volumes/GoogleDrive/My Drive/HAMI/Production/Application/output/Sentiments.xlsx")

                            col1,col2,col3=st.columns(3)

                            #Word cloud overall text
                            topic_corpus = topic_text.values.tolist()
                            #topic_corpus = ' '.join(map(str, topic_corpus)) 

                            #topic_corpus_words = list(sent_to_words(topic_corpus))
                            
                            st.subheader('Word cloud on selected column')

                            comment_words=" "
                            stopwords=stopwords

                            #Iterate through csv file
                            for val in topic_corpus:

                                #typecaste each val to string
                                val=str(val)

                                #split the vale
                                tokens=val.split(maxsplit=2)

                                #Converts each token into lowercase
                                for i in range(len(tokens)):
                                    tokens[i]=tokens[i].lower()

                                comment_words+=" ".join(tokens)+" "
                            #width=1600,height=800, 
                            wordcloud=WordCloud(background_color="White",
                            stopwords=stopwords,min_font_size=10).generate(comment_words)

                            #ploting the word cloud overall texts
                            fig1=plt.figure(figsize=(8,8),facecolor=None)
                            plt.imshow(wordcloud,interpolation='bilinear')
                            plt.axis("off")
                            plt.tight_layout(pad=0)
                            with col1:
                                st.write('**WC on overall input**')
                                st.pyplot(fig1)

                            #
                            pos_comment_words=" "
                            #Iterate through csv file
                            for val in data.pos_keywords:

                                #typecaste each val to string
                                val=str(val)

                                #split the vale
                                tokens=val.split(maxsplit=2)

                                #Converts each token into lowercase
                                for i in range(len(tokens)):
                                    tokens[i]=tokens[i].lower()

                                pos_comment_words+=" ".join(tokens)+" "
                            #width=1600,height=800, 
                            wordcloud=WordCloud(background_color="White",
                            stopwords=stopwords,min_font_size=10).generate(pos_comment_words)

                            #ploting the word cloud overall texts
                            fig2=plt.figure(figsize=(8,8),facecolor=None)
                            plt.imshow(wordcloud,interpolation='bilinear')
                            plt.axis("off")
                            plt.tight_layout(pad=0)
                            with col2:
                                st.write('**WC on Positive keywords**')
                                st.pyplot(fig2)

                          #
                            neg_comment_words=" "
                            #Iterate through csv file
                            for val in data.neg_keywords:

                                #typecaste each val to string
                                val=str(val)

                                #split the vale
                                tokens=val.split(maxsplit=2)

                                #Converts each token into lowercase
                                for i in range(len(tokens)):
                                    tokens[i]=tokens[i].lower()

                                neg_comment_words+=" ".join(tokens)+" "
                            #width=1600,height=800, 
                            wordcloud=WordCloud(background_color="White",
                            stopwords=stopwords,min_font_size=10).generate(neg_comment_words)

                            #ploting the word cloud overall texts
                            fig3=plt.figure(figsize=(8,8),facecolor=None)
                            plt.imshow(wordcloud,interpolation='bilinear')
                            plt.axis("off")
                            plt.tight_layout(pad=0)
                            with col3:
                                st.write('**WC on negative keywords**')
                                st.pyplot(fig3)        


                                

                            sent_count = data['vader_prediction'].value_counts()
                            #sent_count = sent_count[:4,]
                            plt.figure(figsize=(10,5))
                            sns.barplot(sent_count.index, sent_count.values, alpha=0.8)
                            plt.title('Sentiment count across chat data')
                            plt.ylabel('Number of Occurrences', fontsize=12)
                            plt.xlabel('Sentiment polarity', fontsize=12)
                            sent=plt.show()
                            st.pyplot(sent)

                            #data=pd.read_excel('/Volumes/GoogleDrive/My Drive/HAMI/Production/Application/output/Sentiments.xlsx')
                            #data=data[["clean_text","vader_prediction","neg_keywords","pos_keywords"]]
                            #st.dataframe(data)
                            #Export to excel
                            towrite = io.BytesIO()
                            downloaded_file = data.to_excel(towrite, encoding='utf-8', index=False, header=True)
                            towrite.seek(0)  # reset pointer
                            b64 = base64.b64encode(towrite.read()).decode()  # some strings
                            linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Sentiment_Prediction.xlsx">Download Sentiment_predictions file</a>'
                            st.markdown(linko, unsafe_allow_html=True)
                
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
    st.sidebar.info("Analytix Online Singapore")
    st.sidebar.subheader("By")
    st.sidebar.text("© Analytix Online 2021")
        




if __name__== '__main__':
    main()