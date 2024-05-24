from gensim.models import LdaModel
from gensim import corpora
from gensim.corpora import BleiCorpus
import gensim

from nltk.stem.wordnet import WordNetLemmatizer
import nltk

import re
import pymysql
from openpyxl import Workbook

num_words = 30

# # carbon的分类，最好是7   加密货币走向碳中和：https://www.sohu.com/a/469656016_100217347 碳中和相关的关键词lda分析，数量：4359
lda_num_topics = 19 #6 #4
dictionary_path = "models/test/dictionary"+str(lda_num_topics)+".dict"
corpus_path = "models/test/corpus"+str(lda_num_topics)+".lda-c"
lda_model_path = "models/test/lda_model_"+str(lda_num_topics)+"_topics.lda"

dictionary = corpora.Dictionary.load(dictionary_path)
corpus = corpora.BleiCorpus(corpus_path)
lda = LdaModel.load(lda_model_path)
# 对每个文档，得到一个主题分布列表，其中每个元组中的两个元素分别代表主题id和其对应的权重
# 然后获取数据集中所有的主题分布，并最终计算各主题的概率分布
weight_dict = {}
count_dict = {}
for i in range(len(corpus)):
    a = lda[corpus[i]]
    for j in range(len(a)):
        if (a[j][0] not in weight_dict):
            weight_dict[a[j][0]] = a[j][1]
            count_dict[a[j][0]] = 1
        else:
            weight_dict[a[j][0]] += a[j][1]
            count_dict[a[j][0]] += 1
total = 0
for key in weight_dict:
    total += weight_dict[key]
for key in weight_dict:
    weight_dict[key] = weight_dict[key] / total
    print("key:", key, " weight:", weight_dict[key])

for i, topic in enumerate(lda.show_topics(num_topics=lda_num_topics, num_words=num_words)):
    print('#%i: %s' % (i, str(topic)))
