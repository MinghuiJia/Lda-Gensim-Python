# 基于Gensim的LDA模型
## 数据来源
  - data/crawlerdb.sql 中存储着爬取quora的回答文本
  - 导入数据之后，quora_answers_questions表中可以获取对应的answer_content
  - 每个回答answer_content都会作为一个document进行数据预处理，并完成LDA模型训练
## 模型
  - 文本数据预处理：
    - 回答内容拆分成句子
    - 句子拆分成单词（token）
    - 删除拆分后的停用词
    - 词性标注，过滤掉所有非名词的单词
    - 词性还原，还原为词根形式
  - 将处理后的所有数据构建为语料库词典
  - 基于语料库词典，将文本构建为词向量
  - 使用Gensim SDK完成LDA模型的训练
## 其他
- 停用词（data\stopwords.txt）：来源于康奈尔大学实验性SMART信息检索系统创建
- 代码学习reference：https://vladsandulescu.com/topic-prediction-lda-user-reviews/
- LDACluster_copy.py - 使用sklearn的LatentDirichletAllocation模型完成LDA训练的demo文件
  - data\stop_words_old.txt - 是demo对应使用的停用词
  - data\test_data2.txt - 是demo的数据
- 目前代码还差一个基于当前语料库，根据主题连通性（topic coherence）来确定主题中高分词之间的语义相似程度，从而搜索最佳主题，确定语料库中主题的数量