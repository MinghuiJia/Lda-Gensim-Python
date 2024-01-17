from gensim.models import LdaModel
from gensim import corpora


dictionary_path = "models/dictionary.dict"
corpus_path = "models/corpus.lda-c"
lda_num_topics = 10
lda_model_path = "models/lda_model_10_topics.lda"

dictionary = corpora.Dictionary.load(dictionary_path)
corpus = corpora.BleiCorpus(corpus_path)
lda = LdaModel.load(lda_model_path)

for i, topic in enumerate(lda.show_topics(num_topics=lda_num_topics)):
    print('#%i: %s' % (i, str(topic)))
