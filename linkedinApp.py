'''
---------
Import libraries
---------
'''
#Import dash libraries
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from pandas_datareader import data as web
from datetime import datetime as dt
import base64

import flask
import glob
import os
from random import randint

#Import libraries for data manipulation
import re
import numpy as np
import pandas as pd
from pprint import pprint
import string
import pickle

# Import Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Import Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import re, nltk, spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

# Import Sklearn for LDA model
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances

'''
------------
Data import
------------
'''
#Start importing data
lda_model = pickle.load(open('lda_model.sav', 'rb'));
googlesum = pd.read_csv('Datasets/googleDatasciclean.csv');
linkedinSummary = pd.read_csv('Datasets/linkedinDataclean.csv');

#delete the first column of each data set
del googlesum['Unnamed: 0'];
del linkedinSummary['Unnamed: 0'];

'''
----------
Define Functions and Teach Count Vec
----------
'''
#Put linkedin summaries to list
linkedinSummarytwo = linkedinSummary.summary.values.tolist()

#define nlp and spacy
nlp = spacy.load('en', disable=['parser', 'ner'])

#Define function to preprocess data using gensim
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

#Preprocess linkedinSum
linkedinSum = list(sent_to_words(linkedinSummarytwo))

#Define function to change words to its root using lemmatization
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

#Lemmatize linkedinSum words
data_lemmatized = lemmatization(linkedinSum, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#Define vectorizer to create vector of words and remove stop words
vectorizer = CountVectorizer(analyzer='word',
                             min_df=2,                        # minimum reqd occurences of a word
                             max_df=30,
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

#IMPORTANT: teach count vec to fit transform data
data_vectorized = vectorizer.fit_transform(data_lemmatized)
lda_output = lda_model.fit_transform(data_vectorized)

'''
-----------
Keywords Dataframe
-----------
'''
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=10):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]

'''
-----------
Topic Prediction For Summaries
-----------
'''

# Define function to predict topic for a given text document
def predict_topic(text, nlp=nlp):
    global sent_to_words
    global lemmatization

    # Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))

    # Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Step 3: Vectorize transform
    mytext_4 = vectorizer.transform(mytext_3)

    # Step 4: LDA Transform
    topic_probability_scores = lda_model.transform(mytext_4)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()
    return topic, topic_probability_scores

# Predict the topic
text = [googlesum['summary'][2]]
text2 = [googlesum['summary'][15]]
text3 = [googlesum['summary'][23]]
#text = ['skilled in leadership and data science project managment']
topic, prob_scores = predict_topic(text = text)
topic2, prob_scores2 = predict_topic(text = text2)
topic3, prob_scores3 = predict_topic(text = text3)

#topic

'''
----------
Predict Topics for Job Description
----------
'''
#Define lda output
#nlp = spacy.load('en', disable=['parser', 'ner'])

def similar_documents(text, doc_topic_probs, documents = linkedinSum, nlp=nlp, top_n=5, verbose=False):
    topic, x  = predict_topic(text)
    dists = euclidean_distances(x.reshape(1, -1), doc_topic_probs)[0]
    doc_ids = np.argsort(dists)[:top_n]
    if verbose:
        print("Topic KeyWords: ", topic)
        print("Topic Prob Scores of text: ", np.round(x, 1))
        print("Most Similar Doc's Probs:  ", np.round(doc_topic_probs[doc_ids], 1))
    return doc_ids, np.take(documents, doc_ids)

# Get similar documents to job description
jobDescription = pd.read_csv('jobdescription.csv');
jobtext = [jobDescription.JobDescription[0]];

doc_ids, docs = similar_documents(text=jobtext, doc_topic_probs=lda_output, documents = linkedinSum, top_n=1, verbose=True)
#print('\n', docs[0][:500])

'''
-----------
Build Linkedin Profile Recommendation Profiles
-----------
'''
#Import image directory to give Linkedin profile suggestions
image_directory = '/Users/miguelbriones/Desktop/Insight/LinkedinData/LinkedinImages'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'

'''
-----------
Build Dash Application
-----------
'''
# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'
server = flask.Flask(__name__)
#server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server)

app.config.supress_callback_exceptions = True

#load in linkedin logo
image_filename = 'linkedinlogo.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

#load in profile images
profileone_filename = 'LinkedinImages/AbhilashMittapalli.png' # replace with your own image
encodedprofile_imageone = base64.b64encode(open(profileone_filename, 'rb').read())
profileone = 'data:image/png;base64,{}'.format(encodedprofile_imageone.decode())

profiletwo_filename = 'LinkedinImages/ChristianHoward.png' # replace with your own image
encodedprofile_imagetwo = base64.b64encode(open(profiletwo_filename, 'rb').read())
profiletwo = 'data:image/png;base64,{}'.format(encodedprofile_imagetwo.decode())

profilethree_filename = 'LinkedinImages/DougKelly.png' # replace with your own image
encodedprofile_imagethree = base64.b64encode(open(profilethree_filename, 'rb').read())
profilethree = 'data:image/png;base64,{}'.format(encodedprofile_imagethree.decode())


#start app layout / begin div stemming
app.layout = html.Div(children=[
    html.Div([
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                style={'display': 'block', 'margin-left': '80', 'margin-right': '20', 'width': '12%'}),
              ],),
          ], style={'align':'center'}),

    #break in the layout
    html.Br(),
    html.Br(),

#div for dropdown menu
    html.Div([
        dcc.Dropdown(
                id='dropdown', options=[
                {'label': i, 'value': i} for i in jobDescription.JobID.unique()
                ], placeholder='Filter by Job ID'),
                ], style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'width': '40%'}),

    html.Br(),
    html.Br(),

    #holder for image
    html.Div([
        html.Img(id='image', style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'width': '20%'}),
      ],),

    #Div for graph placement
    html.Div([
      dcc.Graph(
        id='my-graph',
        #remove the mode bar
        config={
        'displayModeBar': False }),
      ]),
])

#start the server call back for graph update
@app.callback(Output('my-graph', 'figure'), [Input('dropdown', 'value')])
def update_graph(selected_dropdown_value):
    if selected_dropdown_value == jobDescription.JobID[0]:
        return {'data': [
                    {'x': ['Leadership', 'Tech Savy', 'Academic' ],
                     'y': [prob_scores[0,0], prob_scores[0,1], prob_scores[0,2]], 'type': 'bar' , 'name': 'CD1' },
                    {'x': ['Leadership', 'Tech Savy', 'Academic' ],
                     'y': [0.1, 0.6, 0.3], 'type': 'bar' , 'name': 'JB1' },],
                'layout': {
                     'title': 'Business Traits'}}

    elif selected_dropdown_value == jobDescription.JobID[1]:
        return {'data': [
                    {'x': ['Leadership', 'Tech Savy', 'Academic' ],
                     'y': [prob_scores2[0,0], prob_scores2[0,1], prob_scores2[0,2]], 'type': 'bar', 'name': 'CD2' },
                    {'x': ['Leadership', 'Tech Savy', 'Academic' ],
                     'y': [0.1, 0.7, 0.3], 'type': 'bar' , 'name': 'JB2' },],
                'layout': {
                    'title': 'Business Traits'}}

    else:
        return {'data': [
                    {'x': ['Leadership', 'Tech Savy', 'Academic' ],
                     'y': [prob_scores3[0,0], prob_scores3[0,1], prob_scores3[0,2]], 'type': 'bar', 'name': 'CD3' },
                    {'x': ['Leadership', 'Tech Savy', 'Academic' ],
                     'y': [0.2, 0.6, 0.2], 'type': 'bar' , 'name': 'JB3' },],
                'layout': {
                    'title': 'Business Traits'}}

#server call back for profile image update
@app.callback(dash.dependencies.Output('image', 'src'), [dash.dependencies.Input('dropdown', 'value')])
def update_image_src(selected_dropdown_value):
    if selected_dropdown_value == jobDescription.JobID[0]:
        return profileone

    if selected_dropdown_value == jobDescription.JobID[1]:
        return profiletwo

    else:
        return profilethree

#Run the server
if __name__ == '__main__':
    app.server.run()
