from gensim.models import LdaModel
from gensim import corpora

lda_num_topics = 10
dictionary_path = "models/dictionary"+str(lda_num_topics)+".dict"
corpus_path = "models/corpus"+str(lda_num_topics)+".lda-c"
lda_model_path = "models/lda_model_"+str(lda_num_topics)+"_topics.lda"

dictionary = corpora.Dictionary.load(dictionary_path)
corpus = corpora.BleiCorpus(corpus_path)
lda = LdaModel.load(lda_model_path)

for i, topic in enumerate(lda.show_topics(num_topics=lda_num_topics)):
    print('#%i: %s' % (i, str(topic)))
