# NLP

## Task5 AIWriter

### 数据源

* arXiv dataset —— abstract
* 以此为依托尝试方法，逐渐搜集计算机论文

### 实现

* doc2vec 文本表示
  * 以单句为单位分割做TaggedDocument
  * 以整体段落做分割做TaggedDocument

* 英文文本的去停、去虚
* 尝试lstm，lstm+attention机制

### 问题

* 对AIWriter的理解
* 机器限制
* 句子构建仅仅使用doc2vec是否有更好的方法
*  常规lstm以单词去理解预测，以句子应该怎么理解，是最后的h(n+1)去比对句还是变成词

