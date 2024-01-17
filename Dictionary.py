from gensim import corpora
class Dictionary(object):
    def __init__(self, contents_filtered_words, dictionary_path):
        self.contents_filtered_words = contents_filtered_words
        self.dictionary_path = dictionary_path

    def build(self):
        # 根据语料生成字典
        dictionary = corpora.Dictionary(self.contents_filtered_words)

        # ictionary.dfs,dictionary.token2bow---->用来查看 词语-id,和id->频数

        # 过滤词汇，保留出现频率前keep_n的单词
        dictionary.filter_extremes(keep_n=10000)
        # 词典变得紧凑
        dictionary.compactify()
        corpora.Dictionary.save(dictionary, self.dictionary_path)

        print("corpus dictionary finished... ")
        return dictionary
