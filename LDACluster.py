# -*- coding: utf-8 -*-

import nltk
from nltk.stem.wordnet import WordNetLemmatizer

import gensim
from gensim import corpora
from gensim.corpora import BleiCorpus
from gensim.models.coherencemodel import CoherenceModel

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import re
# import pyLDAvis.gensim

from Dictionary import Dictionary

import collections

import pymysql

from openpyxl import Workbook

def write_tuples_to_excel(tuples, filename):
    workbook = Workbook()
    sheet = workbook.active

    for row in tuples:
        sheet.append(row)

    workbook.save(filename)

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

class LDAClustering():
    def __init__(self, stopwords_path, sql):
        self.sql = sql
        self.content = self.get_contents()
        self.contentsFilteredWords = self.pre_process_corpus(stopwords_path)

    def load_stopwords(self, stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def get_contents(self):
        contents = queryDataFromMysql(self.sql)
        return contents

    def get_words_freq(self, excel_path):
        # 获得词频
        cnt = collections.Counter()  # 创建Counter这个类
        for sentence in self.contentsFilteredWords:
            for word in sentence:
                cnt[word] += 1

        words_freq = cnt.most_common()
        # 将数据写入Excel文件
        write_tuples_to_excel(words_freq, excel_path)

    def pre_process_corpus(self, stopwords_path):
        """
        数据预处理，包括将帖子回答拆分成句子-单词，去掉停用词，过滤掉所有非名词单词，词形还原。
        :param stopwords_path: 停用词路径
        :return: 所有content经过处理后的关键词列表
        """
        print("corpus data preprocessing start... ")

        contents = self.content
        print("total "+str(len(contents)), " contents...")

        # 使用 WordNetLemmatizer 查找每个名词的词根形式
        lem = WordNetLemmatizer()

        # 获取停用词
        stopwords = self.load_stopwords(stopwords_path)

        # 预处理过后的每个回复的有用名词单词
        contentsFilteredWords = []

        # 得到回复帖子之后，将回答拆分成句子，删除停用词，为所有剩余单词标记提取词性标签
        i = 0
        for content in contents:
            print(str(i+1)+" content processing...")
            words = []
            # 回答拆分成句子
            sentences = nltk.sent_tokenize(content.lower())

            # 每个句子中拆分成单词，同时去掉停用词
            for sentence in sentences:
                tokens = nltk.word_tokenize(sentence)
                text = [word for word in tokens if word not in stopwords]
                # 词性标注
                tagged_text = nltk.pos_tag(text)

                for word, tag in tagged_text:
                    words.append({"word": word, "pos": tag})

            # 过滤掉所有非名词单词
            nouns = []
            words = [word for word in words if word["pos"] in ["NN", "NNS"]]

            # 词形还原
            # 将一个单词的各种变体（例如时态、语态、数等）还原为其基本词形或词根形式
            for word in words:
                convert_word = lem.lemmatize(word["word"])
                # 清洗excel中的非法字符，都是不常见的不可显示字符，例如退格，响铃等
                ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
                if (next(ILLEGAL_CHARACTERS_RE.finditer(convert_word), None)):
                    continue
                # 词性还原后再根据停用词筛选一遍
                if (convert_word not in stopwords):
                    nouns.append(convert_word)

            contentsFilteredWords.append(nouns)

            i += 1
        print("corpus data preprocessing finished... ")
        return contentsFilteredWords

    def train(self, lda_model_path, corpus_path, num_topics, id2word, text):
        """
        LDA模型训练
        :param lda_model_path: lda模型训练后的模型文件路径
        :param corpus_path: 语料库向量化后保存的文件路径
        :param num_topics: 主题个数
        :param id2word: 语料库词典
        :return: 所有content经过处理后的关键词列表
        """

        corpus = corpora.BleiCorpus(corpus_path)
        print("corpus vector data loaded... ")
        # for each in corpus:
        #     print(each)
        print("LDA model start train... ")
        lda = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=id2word)
        print("LDA model finished train... ")
        print("LDA model " + str(num_topics) + " topics... ")
        for i, topic in enumerate(lda.show_topics(num_topics=num_topics)):
            print('#%i: %s' % (i, str(topic)))

        # 计算困惑度
        perp = lda.log_perplexity(corpus)
        perpCorrect = np.exp2(-perp)
        # perpCorrect = perp
        print('Perplexity Score: ', perpCorrect)  # 越高越好

        # 计算一致性
        print("calculate LDA model topic coherence...")
        lda_cm = CoherenceModel(model=lda, texts=text, dictionary=id2word, coherence='u_mass', corpus=corpus)
        coherence_lda = lda_cm.get_coherence()
        print('Coherence Score: ', coherence_lda)  # 越高越好

        lda.save(lda_model_path)
        print("LDA model saved... ")
        return lda, coherence_lda, perpCorrect

    def lda(self, dictionary_path, corpus_path, lda_model_path, lda_num_topics):
        """
        LDA主题模型
        :param dictionary_path: 语料库构建的字典路径
        :param corpus_path: 语料库向量化后保存的文件路径
        :param lda_model_path: lda模型训练后的模型文件路径
        :param lda_num_topics: 主题个数
        :return:
        """
        # 获取经过处理后的每个回复的关键词数组
        contentsFilteredWords = self.contentsFilteredWords

        # 创建词典
        dictionary = Dictionary(contentsFilteredWords, dictionary_path).build()

        # 序列化语料库，存储到磁盘
        # 执行serialize时会执行迭代完成向量化
        corpus_memory_friendly = Corpus(contentsFilteredWords, dictionary, corpus_path).serialize()

        # print(corpus_memory_friendly)
        #
        # for vector in corpus_memory_friendly:
        #     print(vector)

        ldamodel, coherence, perp = self.train(lda_model_path, corpus_path, lda_num_topics, dictionary, contentsFilteredWords)

        return ldamodel, coherence, perp


if __name__ == '__main__':
    # import nltk
    #
    # nltk.download()
    #
    # 'wordnet'  'omw-1.4'  'average_perceptron_tagger'  'punkt'

    # # 碳中和相关的关键词lda分析，数量：4359
    sql = '''
                    select answer_content
                    from quora_answers_questions_filter_more
                    where keyword = 'china carbon neutrality'
                    or keyword = 'china double carbon plan'
                    or keyword = 'china carbon peak'
                    ;
                '''
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
    #                                 select answer_content
    #                                 from quora_answers_questions_filter_more
    #                                 where keyword = 'china technology'
    #                                 ;
    #                             '''

    stopwords_path = "data/stopwords.txt"

    topic = []
    perplexity_values = []
    coherence_values = []
    model_list = []
    LDA = LDAClustering(stopwords_path, sql)
    # 获取词频
    LDA.get_words_freq('models_energy/freq_energy.xlsx')
    for i in range(20):
        lda_num_topics = (i+1)
        print("start "+str(lda_num_topics)+" topic lda model train...")

        dictionary_path = "models_carbon/dictionary"+str(lda_num_topics)+".dict"
        corpus_path = "models_carbon/corpus"+str(lda_num_topics)+".lda-c"
        lda_model_path = "models_carbon/lda_model_"+str(lda_num_topics)+"_topics.lda"

        # dictionary_path = "models_energy/dictionary" + str(lda_num_topics) + ".dict"
        # corpus_path = "models_energy/corpus" + str(lda_num_topics) + ".lda-c"
        # lda_model_path = "models_energy/lda_model_" + str(lda_num_topics) + "_topics.lda"

        # dictionary_path = "models_technology/dictionary" + str(lda_num_topics) + ".dict"
        # corpus_path = "models_technology/corpus" + str(lda_num_topics) + ".lda-c"
        # lda_model_path = "models_technology/lda_model_" + str(lda_num_topics) + "_topics.lda"



        # 该代码没有设置lda模型训练时候的迭代次数，默认50次
        ldamodel, coherence, perp = LDA.lda(dictionary_path, corpus_path, lda_model_path, lda_num_topics)

        topic.append(lda_num_topics)
        perplexity_values.append(perp)
        coherence_values.append(coherence)
        model_list.append(ldamodel)

        print("finished " + str(lda_num_topics) + " topic lda model train...")

    # 绘制困惑度和一致性折线图
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(topic, perplexity_values, marker='o')
    plt.title('主题建模-困惑度')
    plt.xlabel('主题数目')
    plt.ylabel('困惑度大小')
    # xticks(np.linspace(1,num_topics,num_topics,endpoint=True))

    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(topic, coherence_values, marker='o')
    plt.title("主题建模-一致性")
    plt.xlabel("主题数目")
    plt.ylabel("一致性大小")

    plt.show()

    # # 可视化
    # vis_data=pyLDAvis.gensim.prepare(model_list[6],corpus,dictionary)
    # pyLDAvis.show(vis_data,open_browser=False)
    # # 保存到本地html
    # pyLDAvis.save_html(vis_data, 'pyLDAvis.html')