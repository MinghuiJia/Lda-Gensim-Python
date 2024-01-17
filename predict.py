import nltk
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import LdaModel
from gensim import corpora

class Predict(object):
    def __init__(self, dictionary_path, lda_model_path, stopwords_path):
        self.dictionary = corpora.Dictionary.load(dictionary_path)
        self.lda = LdaModel.load(lda_model_path)
        self.stopwords_path = stopwords_path

    def load_stopwords(self, stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def extract_lemmatized_nouns(self, new_content):
        stopwords = self.load_stopwords(self.stopwords_path)
        words = []

        sentences = nltk.sent_tokenize(new_content.lower())
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            text = [word for word in tokens if word not in stopwords]
            tagged_text = nltk.pos_tag(text)

            for word, tag in tagged_text:
                words.append({"word": word, "pos": tag})

        lem = WordNetLemmatizer()
        nouns = []
        for word in words:
            if word["pos"] in ["NN", "NNS"]:
                nouns.append(lem.lemmatize(word["word"]))

        return nouns

    def run(self, new_content):
        nouns = self.extract_lemmatized_nouns(new_content)
        new_content_bow = self.dictionary.doc2bow(nouns)
        new_content_lda = self.lda[new_content_bow]

        print (new_content_lda)


if __name__ == '__main__':
    new_content = "It's like eating with a big Italian family. " \
                 "Great, authentic Italian food, good advice when asked, and terrific service. " \
                 "With a party of 9, last minute on a Saturday night, we were sat within 15 minutes. " \
                 "The owner chatted with our kids, and made us feel at home. " \
                 "They have meat-filled raviolis, which I can never find. " \
                 "The Fettuccine Alfredo was delicious. We had just about every dessert on the menu. " \
                 "The tiramisu had only a hint of coffee, the cannoli was not overly sweet, " \
                 "and they had this custard with wine that was so strangely good. " \
                 "It was an overall great experience!"

    stopwords_path = "data/stopwords.txt"
    dictionary_path = "models/dictionary.dict"
    lda_model_path = "models/lda_model_10_topics.lda"

    predict = Predict(dictionary_path, lda_model_path, stopwords_path)
    predict.run(new_content)