# 自然语言处理相关资源合集
本仓库用于存放我个人所需要的和收集的自然语言处理以及部分其他领域的资源合集。

## 书籍、课程与笔记
- [**Deep Learning Book**](https://www.deeplearningbook.org)
- [**Deep Learning Book中文版**](https://github.com/exacity/deeplearningbook-chinese)
- [**Learning from Data笔记**](https://github.com/Doraemonzzz/Learning-from-data)
- [**Python进阶中文版**](https://github.com/eastlakeside/interpy-zh)
- [**Speech and Language Processing第三版**](https://web.stanford.edu/~jurafsky/slp3/)
- [**卡内基梅隆大学2019春 CS 11-747**](http://phontron.com/class/nn4nlp2019/index.html)
- [**斯坦福大学2019春 CS224n**](http://web.stanford.edu/class/cs224n/)
- [**神经网络与深度学习**](https://nndl.github.io) - 复旦邱锡鹏老师编写的教材
- [**NLP研究入门之道**](https://github.com/zibuyu/research_tao) - 清华刘知远老师的NLP科研指导
- [**统计学习方法笔记与代码实现**](https://github.com/fengdu78/lihang-code)

## 代码、工具与项目
### 入门教程
- [**nlp-tutorial**](https://github.com/lyeoni/nlp-tutorial) - 基于PyTorch的深度学习NLP入门代码
- [**NLP-Projects**](https://github.com/gaoisbest/NLP-Projects) - NLP多种类型项目合集
- [**Pandas cookbook**](https://github.com/jvns/pandas-cookbook) - pandas入门教程
- [**PyTorch Tutorial**](https://github.com/yunjey/pytorch-tutorial) - PyTorch入门教程

### 通用深度学习自然语言处理框架与工具
- [**AllenNLP**](https://allennlp.org) - Allen团队开发的基于PyTorch的NLP框架，其中有包括ELMo等多种模型实现
- [**fastNLP**](https://github.com/fastnlp/fastNLP) - 复旦大学深度学习自然语言处理框架
- [**flair**](https://github.com/zalandoresearch/flair) - 轻量级NLP框架
- [**Jiagu**](https://github.com/ownthink/Jiagu) - 基于BiLSTM实现的中文深度学习自然语言处理工具，提供多种中文信息处理基本功能
- [**Kashgari**](https://github.com/BrikerMan/Kashgari) - 基于Keras的精简自然语言处理框架，可用于文本标注与文本分类任务，值得一试
- [**PyText**](https://github.com/facebookresearch/pytext) - Facebook开源的基于PyTorch的自然语言处理框架
- [**StanfordNLP**](https://stanfordnlp.github.io/stanfordnlp/index.html) - 斯坦福大学基于PyTorch开发的自然语言处理工具，可用于分词、序列标注等任务，Stanford CoreNLP的深度学习版替代品
- [**tensor2tensor**](https://github.com/tensorflow/tensor2tensor) - Google研究团队深度学习模型和相关数据集的集中代码仓库，包含多种NLP模型
- [**tensorflow models**](https://github.com/tensorflow/models) - Google模型存放仓库
- [**TorchNLP**](https://github.com/kolloldas/torchnlp) - 基于PyTorch和TorchText实现的深度学习自然语言处理库

### 情感分析
部分相关内容已另行整理至[ABSAPapers](https://github.com/ZhengZixiang/ABSAPapers)
- [**PyTorch Sentiment Analysis**](https://github.com/bentrevett/pytorch-sentiment-analysis) 基于PyTorch的情感分析教程
 
### 自然语言推断
包括文本匹配、问答等句对任务。
- [**MatchZoo**](https://github.com/NTMC-Community/MatchZoo) - 实现多种深度文本匹配模型的基于TensorFlow的工具包
- [**MatchZoo-py**](https://github.com/NTMC-Community/MatchZoo-py) - 上一项目的PyTorch版本
- [**SPM_toolkit**](https://github.com/lanwuwei/SPM_toolkit) - COLING 2018一篇非常好[句对任务综述](https://arxiv.org/abs/1806.04330)作者提供的句对任务匹配工具包
- [**text_matching**](https://github.com/pengming617/text_matching) 基于TensorFlow的多个语义匹配模型实现

### 文本分类
- [**cnn-text-classification-tf**](https://github.com/dennybritz/cnn-text-classification-tf) - 基于TensorFlow的CNN文本分类
- [**Text-Classification-Models-Pytorch**](https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch) - 基于PyTorch实现多种文本分类模型
- [**TextClassification-Pytorch**](https://github.com/songyingxin/TextClassification-Pytorch) - 基于PyTorch实现BERT前时代文本分类模型
- [**NeuralClassifier**](https://github.com/Tencent/NeuralNLP-NeuralClassifier) - 腾讯开源的基于PyTorch的多模型多类型任务文本分类工具
- [**Chinese-Text-Classification-Pytorch**](https://github.com/649453932/Chinese-Text-Classification-Pytorch) - 开箱即用的基于PyTorch实现的中文文本分类框架


### 预训练语言模型与Transformer
部分相关内容已另行整理至[ATPapers](https://github.com/ZhengZixiang/ATPapers)
- [**BERT-Classification-Tutorial**](https://github.com/Socialbird-AILab/BERT-Classification-Tutorial) - BERT文本分类教程
- [**bert-Chinese-classification-task**](https://github.com/NLPScott/bert-Chinese-classification-task) - BERT中文文本分类实践
- [**BERT-for-Sequence-Labeling-and-Text-Classification**](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification) - BERT序列标注与文本分类实践
- [**bert_language_understanding**](https://github.com/brightmart/bert_language_understanding) - 基于TensorFlow实现BERT模型并用于文本分类
- [**bert_ner**](https://github.com/Kyubyong/bert_ner) - 基于PyTorch的BERT命名实体识别
- [**bert-utils**](https://github.com/terrifyzhao/bert-utils) - 一行代码使用BERT生成句向量用于文本分类、文本相似度计算

### 机器阅读理解
- [**Sogou Machine Reading Comprehension Toolkit**](https://github.com/sogou/SMRCToolkit) - 搜狗开源的基于TensorFlow机器阅读理解工具，包括数据预处理和多种模型实现

### 机器翻译
相关内容已另行整理至[MTPapers](https://github.com/ZhengZixiang/MTPapers)

### 聊天机器人、对话系统与问答系统
- [**BertQA**](https://github.com/ankit-ai/BertQA-Attention-on-Steroids) - 斯坦福大学BertQA实现
- [**ChatBotCourse**](https://github.com/warmheartli/ChatBotCourse) - 自己动手做聊天机器人教程
- [**kbqa**](https://github.com/wavewangyue/kbqa) - 基于知识库的问答系统实现

### 命名实体识别
- [**bert-sequence-tagging**](https://github.com/zhpmatrix/bert-sequence-tagging) - 使用BERT做中文序列标注
- [**BioNER-Progress**](https://github.com/lingluodlut/BioNER-Progress) - 生物医学领域命名实体识别进展跟踪
- [**NCRF++**](https://github.com/jiesutd/NCRFpp) - 经典的深度学习序列标注模型

### 关系抽取
- [**OpenNRE**](https://github.com/thunlp/OpenNRE) - 清华开源的关系抽取框架（原基于TensorFlow，2.0版本已迁移到PyTorch）
- [**OpenNRE-PyTorch**](https://github.com/ShulinCao/OpenNRE-PyTorch) - 上一项目的PyTorch版本

### 知识图谱
- [**lightKG**](https://github.com/smilelight/lightKG) - 他人基于PyTorch和TorchText实现的知识图谱技术框架

### 关键词抽取
- [**keyword_extraction**](https://github.com/AimeeLee77/keyword_extraction) - 使用tfidf、TextRank和word2vec实现中文关键词抽取

### 正则表达式
- [**RegExr**](https://regexr.com/) - 正则表达式在线学习、测试与分析网站
- [**Regex Golf**](https://alf.nu/RegexGolf#accesstoken=W0EXx2_lRAMoEeGUVQBx) - 非常好用的经典正则表达式练习网站

### 统计自然语言处理工具包
- [**Apache OpenNLP**](http://opennlp.apache.org/) - Apache开源的Java统计自然语言处理工具包
- [**CRF++**](https://taku910.github.io/crfpp/) - 条件随机场最好用的实现
- [**Python CRFsuite**](https://github.com/scrapinghub/python-crfsuite) - 条件随机场工具CRFsuite的Python封装
- [**FudanNLP**](https://github.com/FudanNLP/fnlp) - 复旦大学开源的统计自然语言处理工具包
- [**HTK**](http://htk.eng.cam.ac.uk) - 基于马尔可夫模型开发的语音识别工具包
- [**Jieba**](https://github.com/fxsjy/jieba) - 结巴分词是Python最常用中文分词
- [**KenLM**](https://kheafield.com/code/kenlm/) - 统计语言模型工具
- [**LTP**](https://ltp.readthedocs.io/zh_CN/latest/index.html) -  哈工大社会计算与信息检索研究中心开源的统计自然语言处理工具包ji
- [**MALLET**](http://mallet.cs.umass.edu) - 马萨诸塞大学开源的Java统计自然语言处理工具包
- [**NLTK**](http://www.nltk.org) - 针对英文的工具包

- [**Pan Gu Segment**](https://archive.codeplex.com/?p=pangusegment) - 盘古开源中文分词
- [**Stanford CoreNLP**](https://nlp.stanford.edu/software/) - 斯坦福大学开源的统计自然语言处理工具包

### ANN近似最近邻
- [**LSHash**](https://github.com/kayzhu/LSHash) - 局部敏感哈希
- [**NearPy**](https://github.com/pixelogik/NearPy)
- [**Simhash**](https://github.com/leonsim/simhash)
- [**FLANN**](https://github.com/primetang/pyflann)

### 其他常用工具
- [**BeautifulSoup**](https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/) - 爬虫常用的HTML和XML数据提取工具
- [**FuzzyWuzzy**](https://github.com/seatgeek/fuzzywuzzy) - Python模糊匹配和编辑距离计算工具
- [**Pytorch模型训练实用教程**](https://github.com/TingsongYu/PyTorch_Tutorial)
- [**SpeechBrain**](https://speechbrain.github.io) - 基于PyTorch的语音处理工具
- [**wikiextractor**](https://github.com/attardi/wikiextractor) - 维基百科语料抽取工具

## 网站与博客
此处仅提供NLP相关站点，优秀博客请链接[blog.md](https://github.com/ZhengZixiang/nlp_resource/blob/master/blog.md)。
- [**AI研习社**](https://www.yanxishe.com)
- [**NLP Progress**](https://nlpprogress.com/)
- [**NLPJob**](http://www.nlpjob.com)
- [**专知**](https://www.zhuanzhi.ai)
- [**机器之心SOTA模型**](https://www.jiqizhixin.com/sota)
### 相关团队与实验室
- [**Tencent AI Lab**](https://ai.tencent.com/ailab/nlp/)

## 论文集与其他资源集
- [**awesome bert**](https://github.com/Jiakui/awesome-bert) - BERT相关资源集
- [**awesome graph neural network**](https://github.com/nnzhan/Awesome-Graph-Neural-Networks) - GNN资源集
- [**awesome graph classification**](https://github.com/benedekrozemberczki/awesome-graph-classification) - 图分类资源集
- [**awesome kaldi**](https://github.com/YoavRamon/awesome-kaldi) - Kaldi相关资源集
- [**awesome knowledge graph**](https://github.com/shaoxiongji/awesome-knowledge-graph) - 知识图谱论文集
- [**awesome knowledge graph**](https://github.com/husthuke/awesome-knowledge-graph) - 另一知识图谱资源集
- [**awesome law nlp research work**](https://github.com/bamtercelboo/Awesome-Law-NLP-Research-Work) - 法律NLP工作资源集
- [**awesome qa**](https://github.com/seriousmac/awesome-qa) - 问答系统资源集
- [**awesome recsys**](https://github.com/chihming/competitive-recsys) - 推荐系统资源集
- [**awesome relation extraction**](https://github.com/roomylee/awesome-relation-extraction) - 关系抽取资源集
- [**awesome sentiment analysis**](https://github.com/laugustyniak/awesome-sentiment-analysis#papers)
- [**awesome speech**](https://github.com/mxer/awesome-speech) - 语音资源集
- [**awesome text generation**](https://github.com/ChenChengKuan/awesome-text-generation) - 文本生成资源集
- [**awesome chinese nlp**](https://github.com/crownpku/Awesome-Chinese-NLP) - 中文自然语言处理资源集
- [**DL-NLP-Readings**](https://github.com/IsaacChanghau/DL-NLP-Readings) - 他人的自然语言处理论文集
- [**Deep Learning with Electronic Health Record (EHR) Systems**](https://practicalai.me/blog/deep-learning-with-ehr-systems) - 深度学习下电子病历系统研究
- [**Research-Line**](https://github.com/ConanCui/Research-Line) - 他人的知识图谱、异构网络、图嵌入与推荐系统论文集
- [**CDCS - Chinese Data Competitions Solutions**](https://github.com/geekinglcq/CDCS) - 中国数据竞赛优胜解集锦
- [**classical reco papers**](https://github.com/wzhe06/Reco-papers) - 王喆大牛的推荐系统论文、学习资料、业界分享
- [**工业界深度推荐系统与CTR预估必读的论文汇总**](https://zhuanlan.zhihu.com/p/83103245)
- [**RecSys**](https://github.com/mJackie/RecSys) - 推荐系统/计算广告/机器学习/CTR预估资料汇总
- [**transformers resource**](https://github.com/sannykim/transformers) - Transformer相关资源集合
- [**MTPapers**](https://github.com/THUNLP-MT/MT-Reading-List) - 清华大学开源机器翻译必读论文集
- [**NREPpapers**](https://github.com/thunlp/NREPapers) - 清华大学开源关系抽取必读论文集
- [**GNNPapers**](https://github.com/thunlp/GNNPapers) - 清华大学开源图神经网络必读论文集
- [**PLMPapers**](https://github.com/thunlp/PLMpapers) - 清华大学开源预训练语言模型必读论文集
- [**nlp-data-augmenta**](https://github.com/quincyliang/nlp-data-augmentation) - NLP数据增强资源集合
- [**CDCS**](https://github.com/geekinglcq/CDCS) - 中国数据竞赛优胜解集锦

## 中文语料资源建设
- [**ChineseGLUE**](https://github.com/chineseGLUE/chineseGLUE) - 中文语言理解测评基准
- [**cn-radical**](https://github.com/skishore/makemeahanzi) - 提取中文偏旁部首和拼音的工具
- [**ChineseNLP**](https://chinesenlp.xyz) - 滴滴整理的中文数据集与SOTA
- [**ChineseNlpCorpus**](https://github.com/SophonPlus/ChineseNlpCorpus) - 中文自然语言处理数据集收集
- [**Chinese-abbreviation-dataset**](https://github.com/zhangyics/Chinese-abbreviation-dataset) - 中文缩略写数据集
- [**chinese-poetry**](https://github.com/chinese-poetry/chinese-poetry) - 最全中华古诗词数据库
- [**nlp_chinese_corpus**](https://github.com/brightmart/nlp_chinese_corpus) - 中文自然语言处理大规模语料库收集
- [**chinese_chatbot_corpus**](https://github.com/codemayq/chinese_chatbot_corpus) - 中文公开聊天语料库
- [**chinese_popular_new_words**](https://github.com/1data-inc/chinese_popular_new_words) - 壹沓科技中文新词表
- [**CNSD**](https://github.com/zengjunjun/CNSD) - 中文自然语言推理数据集
- [**funNLP**](https://github.com/fighting41love/funNLP) - 一个中文词库与结构化信息、工具大全
- [**hanzi_char_featurizer**](https://github.com/howl-anderson/hanzi_char_featurizer) - 汉字字符特征提取器
- [**LDNOOBW**](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) 多语言低俗词汇库
- [**Make Me a Hanzi**](https://github.com/skishore/makemeahanzi) - 开源中文汉字字形数据
- [**nlp-datasets**](https://github.com/niderhoff/nlp-datasets) - NLP部分数据集集合
- [**QQMusicSpider**](https://github.com/yangjianxin1/QQMusicSpider) - 基于Scrapy的QQ音乐爬虫与49万+语料
- [**sensitive-stop-words**](https://github.com/fwwdn/sensitive-stop-words) - 互联网常用敏感词、停止词库
- [**stopwords**](https://github.com/goto456/stopwords) - 中文常用停用词表
- [**THUOCL**](https://github.com/thunlp/THUOCL) - 清华大学开源的多领域中文词库
- [中文情感分析语料大全](https://www.mlln.cn/2018/10/11/%E4%B8%AD%E6%96%87%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90%E8%AF%AD%E6%96%99%E5%BA%93%E5%A4%A7%E5%85%A8-%E5%B8%A6%E4%B8%8B%E8%BD%BD%E5%9C%B0%E5%9D%80/?tdsourcetag=s_pctim_aiomsg)
