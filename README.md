# 自然语言处理相关资源合集
[toc]

本仓库用于存放我个人所需要的和收集的自然语言处理以及部分其他领域的资源合集。

## 书籍、课程与笔记
- [Deep Learning Book](https://www.deeplearningbook.org)
- [Deep Learning Book中文版](https://github.com/exacity/deeplearningbook-chinese)
- [Learning from Data笔记](https://github.com/Doraemonzzz/Learning-from-data)
- [Python进阶中文版](https://github.com/eastlakeside/interpy-zh)
- [Speech and Language Processing第三版](https://web.stanford.edu/~jurafsky/slp3/)
- [卡内基梅隆大学2019春 CS 11-747](http://phontron.com/class/nn4nlp2019/index.html)
- [斯坦福大学2019春 CS224n](http://web.stanford.edu/class/cs224n/)
- [神经网络与深度学习](https://nndl.github.io) - 复旦邱锡鹏老师编写的教材
- [统计学习方法笔记与代码实现](https://github.com/fengdu78/lihang-code)

## 代码、工具与项目
### 入门教程
- [**nlp-tutorial**](https://github.com/lyeoni/nlp-tutorial) - 基于PyTorch的深度学习NLP入门代码

### 通用深度学习自然语言处理框架与工具
- [**AllenNLP**](https://allennlp.org) - Allen团队开发的基于PyTorch的NLP框架，其中有包括ELMo等多种模型实现
- [**Jiagu**](https://github.com/ownthink/Jiagu) - 基于BiLSTM实现的中文深度学习自然语言处理工具，提供多种中文信息处理基本功能
- [**Kashgari**](https://github.com/BrikerMan/Kashgari) - 基于Keras的精简自然语言处理框架，可用于文本标注与文本分类任务，值得一试
- [**PyText**](https://github.com/facebookresearch/pytext) - Facebook开源的基于PyTorch的自然语言处理框架
- [**StanfordNLP**](https://stanfordnlp.github.io/stanfordnlp/index.html) - 斯坦福大学基于PyTorch开发的自然语言处理工具，可用于分词、序列标注等任务，Stanford CoreNLP的深度学习版替代品
- [**tensor2tensor**](https://github.com/tensorflow/tensor2tensor) - Google研究团队深度学习模型和相关数据集的集中代码仓库，包含多种NLP模型
- [**TorchNLP**](https://github.com/kolloldas/torchnlp) - 基于PyTorch和TorchText实现的深度学习自然语言处理库

### 字词向量
- [**Chinese Word Vectors**](https://github.com/Embedding/Chinese-Word-Vectors) 北京师范大学开源的上百种不同预料不同策略预训练中文词向量
- [**ELMoForManyLangs**](https://github.com/HIT-SCIR/ELMoForManyLangs) - 哈工大开源的不同语言的ELMo预训练词表征
- [**Gensim**](https://radimrehurek.com/gensim/) - 导入和训练word embedding的优秀工具包
- [**Glove**](https://nlp.stanford.edu/projects/glove/) - Glove官方地址
- [**word2vec**](https://github.com/dav/word2vec) - word2vec官方C++原始实现
- [**Tencent Word Embedding**](https://ai.tencent.com/ailab/nlp/embedding.html) - 腾讯大规模中文词向量

### 情感分析
- [**ABSA-PyTorch**](https://github.com/songyouwei/ABSA-PyTorch) - 基于PyTorch的多种方面级情感分析模型实现
- [**PyTorch Sentiment Analysis**](https://github.com/bentrevett/pytorch-sentiment-analysis) 基于PyTorch的情感分析教程
 
### 自然语言推断
包括文本匹配、问答等句对任务。
- [**MatchZoo**](https://github.com/NTMC-Community/MatchZoo) - 实现多种深度文本匹配模型的基于TensorFlow的工具包
- [**MatchZoo-py**](https://github.com/NTMC-Community/MatchZoo-py) - 上一项目的PyTorch版本
- [**SPM_toolkit**](https://github.com/lanwuwei/SPM_toolkit) - COLING 2018一篇非常好[句对任务综述](https://arxiv.org/abs/1806.04330)作者提供的句对任务匹配工具包
- [**text_matching**](https://github.com/pengming617/text_matching) 基于TensorFlow的多个语义匹配模型实现

### 文本分类
- [**cnn-text-classification-tf**](https://github.com/dennybritz/cnn-text-classification-tf) - 基于TensorFlow的CNN文本分类
- [**NeuralClassifier**](https://github.com/Tencent/NeuralNLP-NeuralClassifier) - 腾讯开源的基于PyTorch的多模型多类型任务文本分类工具

### 预训练语言模型与Transformer
- [**bert**](https://github.com/google-research/bert) Google BERT官方代码
- [**transformer-tensorflow**](https://github.com/DongjunLee/transformer-tensorflow) - 基于TensorFlow的Transformer实现
- [**huggingface transformers**](https://github.com/huggingface/transformers) - 强烈推荐，同时支持TensorFlow 2.0和PyTorch和支持多种预训练语言模型工具
- [**BERT-Classification-Tutorial**](https://github.com/Socialbird-AILab/BERT-Classification-Tutorial) - BERT文本分类教程
- [**bert-Chinese-classification-task**](https://github.com/NLPScott/bert-Chinese-classification-task) - BERT中文文本分类实践
- [**BERT-for-Sequence-Labeling-and-Text-Classification**](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification) - BERT序列标注与文本分类实践
- [**bert_language_understanding**](https://github.com/brightmart/bert_language_understanding) - 基于TensorFlow实现BERT模型并用于文本分类
- [**bert_ner**](https://github.com/Kyubyong/bert_ner) - 基于PyTorch的BERT命名实体识别
- [**BERT-pytorch**](https://github.com/codertimo/BERT-pytorch) - 基于PyTorch的BERT官方实现
- [**bert-as-service**](https://github.com/hanxiao/bert-as-service) - 用BERT将变长句子映射成定长向量的工具
- [**bert-utils**](https://github.com/terrifyzhao/bert-utils) - 一行代码使用BERT生成句向量用于文本分类、文本相似度计算
- [**Keras Attention Layer**](https://github.com/thushv89/attention_keras) - 基于Keras的Attention层实现
- [**keras-bert**](https://github.com/CyberZHG/keras-bert) - 基于Keras的BERT实现和加载工具
- [**BERT-keras**](https://github.com/Separius/BERT-keras) - 另一个基于Keras的BERT实现
- [**bert4keras**](https://github.com/bojone/bert4keras) - 苏神版本的基于Keras的BERT实现
- [**GPT-2-Pytorch Text-Generator**](https://github.com/graykode/gpt-2-Pytorch) - GPT-2文本生成实现
- [**GPT2-Chinese**](https://github.com/Morizeyao/GPT2-Chinese) - GPT-2中文训练代码
- [**XLNet**](https://github.com/zihangdai/xlnet) - Google [XLNet](https://arxiv.org/abs/1906.08237)实现
- [**OpenCLaP**](https://github.com/thunlp/OpenCLaP) - 清华大学开源的多领域（民事、刑事、百度百科）中文预训练语言模型仓库
- [**Chinese-BERT-wwm模型**](https://github.com/ymcui/Chinese-BERT-wwm)
- [**XLNet中文预训练模型**](https://github.com/ymcui/Chinese-PreTrained-XLNet)
- [**RoBERTa中文预训练模型**](https://github.com/brightmart/roberta_zh)
- [**ALBERT中文预训练模型**](https://github.com/brightmart/albert_zh)

### 机器阅读理解
- [**Sogou Machine Reading Comprehension Toolkit**](https://github.com/sogou/SMRCToolkit) - 搜狗开源的基于TensorFlow机器阅读理解工具，包括数据预处理和多种模型实现

### 机器翻译
- [**OpenNMT-py**](https://github.com/OpenNMT/OpenNMT-py) - 基于PyTorch的机器翻译框架
- [**OpenNMT-tf**](https://github.com/OpenNMT/OpenNMT-tf) - 上个项目的TensorFlow版本
- [**Subword Neural Machine Translation**](https://github.com/rsennrich/subword-nmt) - 为机器翻译等任务提供切分subword功能的工具

### 聊天机器人、对话系统与问答系统
- [**BertQA**](https://github.com/ankit-ai/BertQA-Attention-on-Steroids) - 斯坦福大学BertQA实现
- [**ChatBotCourse**](https://github.com/warmheartli/ChatBotCourse) - 自己动手做聊天机器人教程
- [**kbqa**](https://github.com/wavewangyue/kbqa) - 基于知识库的问答系统实现

### 关键词抽取
- [**keyword_extraction**](https://github.com/AimeeLee77/keyword_extraction) - 使用tfidf、TextRank和word2vec实现中文关键词抽取

### 正则表达式
- [**RegExr**](https://regexr.com/) - 正则表达式在线学习、测试与分析网站
- [**Regex Golf**](https://alf.nu/RegexGolf#accesstoken=W0EXx2_lRAMoEeGUVQBx) - 非常好用的经典正则表达式练习网站

### 中文资源与工具
- [**cn-radical**](https://github.com/skishore/makemeahanzi) - 提取中文偏旁部首和拼音的工具
- [**funNLP**](https://github.com/fighting41love/funNLP) - 一个中文词库与结构化信息、工具大全
- [**Make Me a Hanzi**](https://github.com/skishore/makemeahanzi) - 开源中文汉字字形数据
- [**THUOCL**](https://github.com/thunlp/THUOCL) - 清华大学开源的多领域中文词库

### 统计自然语言处理工具包
- [**Apache OpenNLP**](http://opennlp.apache.org/) - Apache开源的Java统计自然语言处理工具包
- [**CRF++**](https://taku910.github.io/crfpp/) - 条件随机场最好用的实现
- [**Python CRFsuite**](https://github.com/scrapinghub/python-crfsuite) - 条件随机场工具CRFsuite的Python封装
- [**FudanNLP**](https://github.com/FudanNLP/fnlp) - 复旦大学开源的统计自然语言处理工具包
- [**HTK**](http://htk.eng.cam.ac.uk) - 基于马尔可夫模型开发的语音识别工具包
- [**Jieba**](https://github.com/fxsjy/jieba) - 结巴分词是Python最常用中文分词
- [**KenLM**](https://kheafield.com/code/kenlm/) - 统计语言模型工具
- [**LTP**](https://ltp.readthedocs.io/zh_CN/latest/index.html) -  哈工大社会计算与信息检索研究中心开源的统计自然语言处理工具包
- [**MALLET**](http://mallet.cs.umass.edu) - 马萨诸塞大学开源的Java统计自然语言处理工具包
- [**NLTK**](http://www.nltk.org) - 针对英文的工具包
- [**NiuTrans**](http://www.niutrans.com/niutrans/NiuTrans.ch.html) - 东北大学自然语言处理实验室开源的统计机器翻译工具
- [**Pan Gu Segment**](https://archive.codeplex.com/?p=pangusegment) - 盘古开源中文分词
- [**Stanford CoreNLP**](https://nlp.stanford.edu/software/) - 斯坦福大学开源的统计自然语言处理工具包

### 其他常用工具
- [**BeautifulSoup**](https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/) - 爬虫常用的HTML和XML数据提取工具
- [**FuzzyWuzzy**](https://github.com/seatgeek/fuzzywuzzy) - Python模糊匹配和编辑距离计算工具
- [**Pytorch模型训练实用教程**](https://github.com/TingsongYu/PyTorch_Tutorial)
- [**SpeechBrain**](https://speechbrain.github.io) - 基于PyTorch的语音处理工具
