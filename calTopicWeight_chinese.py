from gensim.models import LdaModel
from gensim import corpora
from gensim.corpora import BleiCorpus
import gensim

from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import jieba
import jieba.posseg as pseg
jieba.enable_paddle()  # 启动paddle模式。 0.40版之后开始支持，早期版本不支持
import re
import pymysql
from openpyxl import Workbook

def write_tuples_to_excel(tuples, filename):
    workbook = Workbook()
    sheet = workbook.active

    for row in tuples:
        sheet.append(row)

    workbook.save(filename)

class Corpus(object):
    def __init__(self, contents_filtered_words, dictionary, corpus_path):
        self.contents_filtered_words = contents_filtered_words
        self.dictionary = dictionary
        self.corpus_path = corpus_path

    def __iter__(self):
        # 对每个文本的word列表转换成稀疏词袋向量
        for content in self.contents_filtered_words:
            yield self.dictionary.doc2bow(content)

        print("corpus iter finished... ")

    def serialize(self):
        # Gensim通过流式语料库接口实现，文件以懒惰的方式从（分别存储到）磁盘读取，一次一个文档，而不是一次将整个语料库读入主存储器
        BleiCorpus.serialize(self.corpus_path, self, id2word=self.dictionary)
        print("corpus serialize finished... ")

        return self

def load_stopwords( stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def queryDataFromMysql(sql, param=None):
    # 存储到数据库
    # 连接数据库
    conn = pymysql.connect(host='localhost', port=3306,
                           user='root', password='JMHjmh1998',
                           database='crawlerdb')

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = conn.cursor()

    try:
        cursor.execute(sql, param)
        contents = cursor.fetchall()
        newContents = [content[0] for content in contents]
        return newContents
    except Exception as e:
        print(e)
        return None
    finally:
        conn.close()

def pre_process_corpus(stopwords_path, sql):
    """
    数据预处理，包括将帖子回答拆分成句子-单词，去掉停用词，过滤掉所有非名词单词，词形还原。
    :param stopwords_path: 停用词路径
    :return: 所有content经过处理后的关键词列表
    """
    print("corpus data preprocessing start... ")

    contents = queryDataFromMysql(sql)
    print("total " + str(len(contents)), " contents...")

    # 使用 WordNetLemmatizer 查找每个名词的词根形式
    lem = WordNetLemmatizer()

    # 获取停用词
    stopwords = load_stopwords(stopwords_path)

    # 预处理过后的每个回复的有用名词单词
    contentsFilteredWords = []
    contentsFilteredWords_one = []

    # 得到回复帖子之后，将回答拆分成句子，删除停用词，为所有剩余单词标记提取词性标签
    i = 0
    for content in contents:
        print(str(i + 1) + " content processing...")
        words = []
        # 回答拆分成句子
        # sentences = nltk.sent_tokenize(content.lower())
        sentences = re.split('(?<!\\w\\.\\w.)(?<![A-Z][a-z]\\.)(?<=\\.|\\?)\\s', content)

        # 每个句子中拆分成单词，同时去掉停用词
        # for sentence in sentences:
        #     tokens = nltk.word_tokenize(sentence)
        #     text = [word for word in tokens if word not in stopwords]
        #     # 词性标注
        #     tagged_text = nltk.pos_tag(text)
        #
        #     for word, tag in tagged_text:
        #         words.append({"word": word, "pos": tag})

        # 每个句子中拆分成单词，同时去掉停用词
        for sentence in sentences:
            # 中文
            tokens = pseg.cut(sentence, use_paddle=True)  # paddle模式

            # 中文
            for word, tag in tokens:
                if (word not in stopwords):
                    words.append({"word": word, "pos": tag})



        # 过滤掉所有非名词单词
        nouns = []
        # words = [word for word in words if word["pos"] in ["NN", "NNS"]]
        words = [word for word in words if
                 word["pos"] in ["n", "f", "s", "t", "nr", "ns", "nt", "nw", "nz", "vn", "PER", "LOC", "ORG"]]

        # 词形还原
        # 将一个单词的各种变体（例如时态、语态、数等）还原为其基本词形或词根形式
        # for word in words:
        #     convert_word = lem.lemmatize(word["word"])
        #     # 清洗excel中的非法字符，都是不常见的不可显示字符，例如退格，响铃等
        #     ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
        #     if (next(ILLEGAL_CHARACTERS_RE.finditer(convert_word), None)):
        #         continue
        #     # 词性还原后再根据停用词筛选一遍
        #     if (convert_word not in stopwords):
        #         nouns.append(convert_word)

        for word in words:
            # 词性还原后再根据停用词筛选一遍
            word_str = word["word"].strip()
            if (word_str not in stopwords):
                nouns.append(word_str)

        contentsFilteredWords.append(nouns)
        contentsFilteredWords_one.extend(nouns)
        i += 1
    print("corpus data preprocessing finished... ")
    return contentsFilteredWords_one

num_words = 30

# # carbon的分类，最好是7   加密货币走向碳中和：https://www.sohu.com/a/469656016_100217347 碳中和相关的关键词lda分析，数量：4359
lda_num_topics = 8 #6 #4
dictionary_path = "models/sijilao/dictionary"+str(lda_num_topics)+".dict"
corpus_path = "models/sijilao/corpus"+str(lda_num_topics)+".lda-c"
lda_model_path = "models/sijilao/lda_model_"+str(lda_num_topics)+"_topics.lda"

# # energy的分类，最好是 7   能源相关的关键词lda分析，数量：6282 俄罗斯的能源对比  海外正在酝酿一个新能源联盟吗?在欧佩克之外，中国和俄罗斯是能源市场的主要参与者吗   为什么沙特阿拉伯、俄罗斯和中国不停止向美国出口东西，没有石油和许多制成品等，美国肯定会失败或被削弱
# lda_num_topics = 7 #3
# dictionary_path = "models_energy/dictionary" + str(lda_num_topics) + ".dict"
# corpus_path = "models_energy/corpus" + str(lda_num_topics) + ".lda-c"
# lda_model_path = "models_energy/lda_model_" + str(lda_num_topics) + "_topics.lda"

# # technology的分类，最好是 10   科技相关的关键词lda分析，数量：2512 semiconductor-半导体 space-太空、航天 手机 ai 制造业-军事、导弹 系统 市场 公司 与日本、美国、印度比较
# lda_num_topics = 10 #2 3 4 10
# keyword_type = 'technology'
# dictionary_path = "models_technology/dictionary"+str(lda_num_topics)+".dict"
# corpus_path = "models_technology/corpus"+str(lda_num_topics)+".lda-c"
# lda_model_path = "models_technology/lda_model_"+str(lda_num_topics)+"_topics.lda"

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

# # 碳中和相关的关键词lda分析，数量：4359
# sql = '''
#                 select answer_content
#                 from quora_answers_questions_filter_more
#                 where keyword = 'china carbon neutrality'
#                 or keyword = 'china double carbon plan'
#                 or keyword = 'china carbon peak'
#                 ;
#             '''
# # 能源相关的关键词lda分析，数量：6282
# sql = '''
#                     select answer_content
#                     from quora_answers_questions_filter_more
#                     where keyword = 'china new energy'
#                     or keyword = 'china energy conservation'
#                     ;
#                 '''
# 科技相关的关键词lda分析，数量：2512
# sql = '''
#                                     select answer_content
#                                     from quora_answers_questions_filter_more
#                                     where keyword = 'china technology'
#                                     ;
#                                 '''
# contentsFilteredWords = pre_process_corpus('data/stopwords.txt', sql)
#
# # 创建词典
# dictionary = corpora.Dictionary([contentsFilteredWords])
# # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
# corpus = [dictionary.doc2bow(text) for text in [contentsFilteredWords]]
# # 使用gensim来创建LDA模型对象
# lda_model = gensim.models.LdaModel(corpus, num_topics=lda_num_topics, id2word=dictionary)
#
# for i in range(len(corpus)):
#     print(lda_model[corpus[i]])
