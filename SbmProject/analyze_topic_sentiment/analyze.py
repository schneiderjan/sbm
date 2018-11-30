from gensim import models
import pickle
from gensim.corpora import Dictionary
import pandas as pd
from textblob import TextBlob

def getTopic(sent_topics_df, review):
    best_word_count = 0
    best_topic = -1
    for index, row in sent_topics_df.iterrows():
        words = row[1].split(',')  
        word_count = 0
        for word in words:
            if review.find(word) != -1:
                word_count = word_count + 1
        if(word_count > best_word_count):
            best_word_count = word_count
            best_topic = row[0]
    
    return best_topic

def getSentiment(review):
    testimonial = TextBlob(review)
    sentiment = testimonial.sentiment
    return sentiment.polarity

print('Load gensim dict.')
dictionary = Dictionary.load('../topic_modelling/dictionary.gensim')
print('Unpickle corpus.')
corpus = pickle.load(open('../topic_modelling/corpus.pkl', 'rb'))
print('Load gensim Lda model.')
ldamodel = models.ldamodel.LdaModel.load('../topic_modelling/lda_model.gensim')

# Load Reviews
reviews_df = pd.read_csv('reviews_rank_for_appID_and_week.csv',
                 infer_datetime_format=True, error_bad_lines=False)

 # Create topics dataframe
sent_topics_df = pd.DataFrame()

for j in range(0,10):
    wp = ldamodel.show_topic(j)
    topic_keywords = ", ".join([word for word, prop in wp])
    sent_topics_df = sent_topics_df.append(pd.Series([int(j), topic_keywords]), ignore_index=True)
    
import csv

with open('analyzed.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for index, reviewRow in reviews_df.iterrows():
        # TODO: THIS SPLIT IS NOT YET COVERING ALL CASES
        reviews = reviewRow['content'].split("', '")
        for review in reviews:  
            topic = getTopic(sent_topics_df, review)
            sentiment = getSentiment(review)
            filewriter.writerow([reviewRow['appID'], reviewRow['week'], review, reviewRow['chart'], reviewRow['rank'], reviewRow['mean_rank'], reviewRow['rank_previous_week'], topic, sentiment])

