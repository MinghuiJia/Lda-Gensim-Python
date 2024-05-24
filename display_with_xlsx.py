from gensim.models import LdaModel
from gensim import corpora
from openpyxl import Workbook

def write_tuples_to_excel(tuples, filename):
    workbook = Workbook()
    sheet = workbook.active

    for row in tuples:
        sheet.append(row)

    workbook.save(filename)

# 修改之后可输出每个主题不同数量的关键词
num_words = 30

# 总体的分类（三组数据）
# lda_num_topics = 7 #8 #9  # 5

# # carbon的分类，最好是7   加密货币走向碳中和：https://www.sohu.com/a/469656016_100217347 碳中和相关的关键词lda分析，数量：4359
# lda_num_topics = 7 #6 #4
# dictionary_path = "models_copy/models_carbon/dictionary"+str(lda_num_topics)+".dict"
# corpus_path = "models_copy/models_carbon/corpus"+str(lda_num_topics)+".lda-c"
# lda_model_path = "models_copy/models_carbon/lda_model_"+str(lda_num_topics)+"_topics.lda"

# # energy的分类，最好是 7   能源相关的关键词lda分析，数量：6282 俄罗斯的能源对比  海外正在酝酿一个新能源联盟吗?在欧佩克之外，中国和俄罗斯是能源市场的主要参与者吗   为什么沙特阿拉伯、俄罗斯和中国不停止向美国出口东西，没有石油和许多制成品等，美国肯定会失败或被削弱
# lda_num_topics = 7 #3
# dictionary_path = "models_copy/models_energy/dictionary" + str(lda_num_topics) + ".dict"
# corpus_path = "models_copy/models_energy/corpus" + str(lda_num_topics) + ".lda-c"
# lda_model_path = "models_copy/models_energy/lda_model_" + str(lda_num_topics) + "_topics.lda"

# # technology的分类，最好是 10   科技相关的关键词lda分析，数量：2512 semiconductor-半导体 space-太空、航天 手机 ai 制造业-军事、导弹 系统 市场 公司 与日本、美国、印度比较
lda_num_topics = 19 #2 3 4 10
dictionary_path = "models/test/dictionary"+str(lda_num_topics)+".dict"
corpus_path = "models/test/corpus"+str(lda_num_topics)+".lda-c"
lda_model_path = "models/test/lda_model_"+str(lda_num_topics)+"_topics.lda"

dictionary = corpora.Dictionary.load(dictionary_path)
corpus = corpora.BleiCorpus(corpus_path)
lda = LdaModel.load(lda_model_path)

# 表头信息
lizt = []
tuples = ["Topic Number"]
for i in range(num_words):
    tuples.append("Topic words "+str(i+1))
    tuples.append("Topic words "+str(i+1)+" weight")
tuples = tuple(tuples)
lizt.append(tuples)

# 解析每个topic的关键词
for i, topic in enumerate(lda.show_topics(num_topics=lda_num_topics, num_words=num_words)):
    topic_num = topic[0]
    topic_keywords_list = topic[1].split(" + ")
    tuples = [i+1]
    for k in range(len(topic_keywords_list)):
        weight_keyword = topic_keywords_list[k].strip("'").split("*")
        weight = weight_keyword[0]
        keyword = weight_keyword[1].strip('"')
        tuples.append(keyword)
        tuples.append(float(weight))
    tuples = tuple(tuples)
    lizt.append(tuples)
    print('#%i: %s' % (i, str(topic)))

# 写入excel
write_tuples_to_excel(lizt, "models/test/topics_"+str(num_words)+"_keywords.xlsx")