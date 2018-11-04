import numpy as np
import pandas as pd
from nltk import corpus
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import models
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
import re
import pickle

dtypes_dict = {'title': str, 'content': str}
conv = {'title': lambda x: str(x)}
df = pd.read_csv('../data/app_reviews_1000.csv', infer_datetime_format=True, error_bad_lines=False)
df = df[['title', 'content']]
stopwords = corpus.stopwords.words('english')


def pre_process(x):
    x = re.sub('[^a-z\s]', '', x.lower())  # lowercase
    x = [w for w in x.split() if w not in set(stopwords)]  # remove stopwords
    # x = [w for w in word_tokenize(x)]  # tokenize
    return ' '.join(x)  # join the list


df['title_pp'] = df['title'].apply(pre_process)
df['title_pp'] = df['title'].apply(word_tokenize)
df['content_pp'] = df['content'].apply(pre_process)
df['content_pp'] = df['content'].apply(word_tokenize)

text_data = df.title_pp.tolist()
text_data.append(df.content_pp.tolist())

# simple_text_data=[]
# for i in text_data:
#     for j in i:
#         simple_text_data.append(j)

dictionary = Dictionary(text_data)
asdasd =[dictionary.doc2bow(text) for text in text_data]

ldamodel = models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)