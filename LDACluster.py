# -*- coding: utf-8 -*-

import nltk
from nltk.stem.wordnet import WordNetLemmatizer

import gensim
from gensim import corpora
from gensim.corpora import BleiCorpus

from Dictionary import Dictionary

import pymysql

def queryDataFromMysql(sql, param=None):
    # 存储到数据库
    # 连接数据库
    conn = pymysql.connect(host='localhost', port=3306,
                           user='root', password='123456',
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
    def load_stopwords(self, stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def get_contents(self):
        sql = '''
                    select answer_content from quora_answers_questions;
                '''
        contents = queryDataFromMysql(sql)
        return contents

    def pre_process_corpus(self, stopwords_path):
        """
        数据预处理，包括将帖子回答拆分成句子-单词，去掉停用词，过滤掉所有非名词单词，词形还原。
        :param stopwords_path: 停用词路径
        :return: 所有content经过处理后的关键词列表
        """
        print("corpus data preprocessing start... ")

        contents = self.get_contents()
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
                nouns.append(lem.lemmatize(word["word"]))

            contentsFilteredWords.append(nouns)

            i += 1
        print("corpus data preprocessing finished... ")
        return contentsFilteredWords

    def train(self, lda_model_path, corpus_path, num_topics, id2word):
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
        lda.save(lda_model_path)
        print("LDA model saved... ")
        return lda

    def lda(self, dictionary_path, corpus_path, lda_model_path, lda_num_topics, stopwords_path='./data/stop_words.txt'):
        """
        LDA主题模型
        :param dictionary_path: 语料库构建的字典路径
        :param corpus_path: 语料库向量化后保存的文件路径
        :param lda_model_path: lda模型训练后的模型文件路径
        :param lda_num_topics: 主题个数
        :param stopwords_path: 停用词路径
        :return:
        """
        # 获取经过处理后的每个回复的关键词数组
        contentsFilteredWords = self.pre_process_corpus(stopwords_path=stopwords_path)

        # 创建词典
        dictionary = Dictionary(contentsFilteredWords, dictionary_path).build()

        # 序列化语料库，存储到磁盘
        # 执行serialize时会执行迭代完成向量化
        corpus_memory_friendly = Corpus(contentsFilteredWords, dictionary, corpus_path).serialize()

        # print(corpus_memory_friendly)
        #
        # for vector in corpus_memory_friendly:
        #     print(vector)

        self.train(lda_model_path, corpus_path, lda_num_topics, dictionary)



if __name__ == '__main__':
    # import nltk
    #
    # nltk.download()
    #
    # 'wordnet'  'omw-1.4'  'average_perceptron_tagger'  'punkt'

    dictionary_path = "models/dictionary.dict"
    stopwords_path = "data/stopwords.txt"
    corpus_path = "models/corpus.lda-c"
    lda_model_path = "models/lda_model_10_topics.lda"
    lda_num_topics = 10

    # 该代码没有设置lda模型训练时候的迭代次数，默认50次
    LDA = LDAClustering()
    LDA.lda(dictionary_path, corpus_path, lda_model_path, lda_num_topics, stopwords_path)