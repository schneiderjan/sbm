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
df = pd.read_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\reviews_rank_for_appID_and_week.csv',
                 infer_datetime_format=True, error_bad_lines=False)
stopwords = corpus.stopwords.words('english')
print('read csv file.')


def pre_process(x):
    x = re.sub('[^a-z\s]', '', x.lower())  # lowercase
    x = [w for w in x.split() if w not in set(stopwords)]  # remove stopwords
    # x = [w for w in word_tokenize(x)]  # tokenize
    return ' '.join(x)  # join the list


print('Start pre processing.')
stuff = df['content']
stuff_2 = stuff.apply(pre_process)
stuff_3 = stuff_2.apply(word_tokenize)
print('Finished pre processing.')

print('Create doc2bow and dictionary.')
dictionary = Dictionary(stuff_3)
print("Created dict with {0}".format(dictionary))
corpus = [dictionary.doc2bow(text) for text in stuff_3]

print('Save corpus and dictionary.')
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

print('Create TF-IDF model.')
tfidf = models.TfidfModel(corpus)
print('Transform corpus to tfidf vector space.')
transformed_corpus = tfidf[corpus]

print('Make LDA model.')
ldamodel = models.ldamodel.LdaModel(transformed_corpus, num_topics=10, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)
