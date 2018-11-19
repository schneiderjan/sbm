import pandas as pd
from nltk import corpus
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim import models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import pickle

dtypes_dict = {'title': str, 'content': str}
conv = {'title': lambda x: str(x)}
df = pd.read_csv('D:\GoogleDriveJads\Projects\JM0170-SBM-gr17\Data\\reviews_rank_for_appID_and_week.csv',
                 infer_datetime_format=True, error_bad_lines=False)

stopwords = corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

extra_stopwords = ['app', 'love', 'use', 'get', 'like', 'great']
stopwords.extend(extra_stopwords)


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# def pre_process(x):
#     x = re.sub('[^a-z\s]', '', x.lower())  # lowercase
#     x = [w for w in x.split() if w not in set(stopwords) and len(w) >= 3]  # remove stopwords
#     x = [w for w in lemmatize_stemming(x)]  # tokenize
#     return ' '.join(x)  # join the list


def pre_process(text):
    """
    Simple pre process does lowercase and tokenize
    then stopwords removed and words greater equal of length 3 are removed
    finally words are lemmatized/stemmed

    :param text: One list of one app row as in the csv file
    :return: a single list of all words for the row that was inserted.
    """
    result = []
    # print(text.strip())
    for token in simple_preprocess(text):
        token = token.replace('\\n', '')
        if token not in set(stopwords) and len(token) >= 3:
            result.append(lemmatize_stemming(token))
    return result


unique_emo = set()


def filter_escape_characters(text):
        for t in text:
            if t.startswith('ud'):
                unique_emo.add(t)

def remove_escape_characters(x):
    return [item for item in x if item not in weird_stopwords]

print('Start pre_processing.')
stuff = df['content']
pre_processed_reviews = []
for app_reviews in stuff:
    pre_processed = pre_process(app_reviews)
    filter_escape_characters(pre_processed)
    pre_processed_reviews.append(pre_processed)
weird_stopwords = list(unique_emo)

print('weird')
# print(pre_processed_reviews)
print(weird_stopwords)
p = []

for reviews in pre_processed_reviews:
    p.append((remove_escape_characters(reviews)))
pre_processed_reviews = p

print('Pickle dump pre processed reviews.')
pickle.dump(pre_processed_reviews, open('pre_processed_reviews.pkl', 'wb'))

print('Create doc2bow and dictionary.')
dictionary = Dictionary(pre_processed_reviews)
print("Created dict with {0}".format(dictionary))
corpus = [dictionary.doc2bow(text) for text in pre_processed_reviews]

print('Save corpus and dictionary.')
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')
