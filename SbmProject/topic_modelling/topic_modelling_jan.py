from gensim import models
import pickle
from gensim.corpora import Dictionary

with open('D:\CodeJads\sbm\SbmProject\data_extraction\\corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)
dictionary = Dictionary.load('D:\CodeJads\sbm\SbmProject\data_extraction\\dictionary.gensim')

print('Create TF-IDF model.')
tfidf = models.TfidfModel(corpus)
print('Transform corpus to tfidf vector space.')
tfidf_corpus = tfidf[corpus]
pickle.dump(tfidf_corpus, open('corpus_tfidf.pkl', 'wb'))

print('Make LDA model.')
ldamodel = models.ldamodel.LdaModel(tfidf_corpus, num_topics=10, id2word=dictionary, passes=15)
ldamodel.save('lda_model.gensim')
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)
