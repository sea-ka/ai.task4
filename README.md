### 1. 核心功能

   - 目标：区分垃圾邮件（标记为1）和普通邮件（标记为0）

   - 方法：使用词频统计 + 多项式朴素贝叶斯（MultinomialNB）分类

   - 输入：151个已标注的邮件文本（0.txt到150.txt）

   - 输出：对新邮件（151.txt到155.txt）进行分类预测

### 2. 关键步骤

#### (1) 文本预处理 (get_words)
   - 读取文件内容

   - 过滤无效字符（标点、数字等）：re.sub(r'[.【】0-9、——。，！~\*]', '', line)

   -  使用结巴分词（jieba.cut）切词

   - 过滤单字词（保留长度>1的词）

#### (2) 构建词库 (get_top_words)

   - 遍历所有邮件文本，统计每个词的出现频率

   - 返回出现频率最高的前100个词（top_words）

#### (3) 特征向量化

   - 对每封邮件，统计top_words中每个词的出现次数，生成词频向量

   - 例如：
```commandline
# 假设 top_words = ["期刊", "论文", "SCI"]
# 某邮件内容为 ["期刊", "SCI", "SCI"]
# 对应的词频向量为 [1, 0, 2]
```
#### (4) 训练分类器

   - 标签：0-126.txt为垃圾邮件（1），127-150.txt为普通邮件（0）

   - 使用MultinomialNB训练模型：
```commandline
model.fit(vector, labels)
```
(5) 预测新邮件 (predict)

   - 对新邮件文本进行相同的预处理和词频统计

   - 使用训练好的模型预测分类结果

### 3. 关键变量说明

|     变量名     |       说明 |
|:-----------:|--|
|  all_words  | 所有邮件的分词结果（二维列表） |
|  top_words  | 最高频的100个词（特征词） |
|   vector    | 所有邮件的词频特征矩阵（形状：151×100） |
|   labels    | 邮件标签（1=垃圾邮件，0=普通邮件） |
|   model     | 训练好的朴素贝叶斯分类器 |

### 4. 示例输出

```commandline
151.txt分类情况:垃圾邮件
152.txt分类情况:普通邮件
153.txt分类情况:垃圾邮件
154.txt分类情况:普通邮件
155.txt分类情况:垃圾邮件
```
### 5. 改进建议

   - 数据增强：增加更多标注数据提升准确性

   - 特征优化：

     - 使用TF-IDF替代简单词频

     - 加入停用词过滤（如"的"、"是"等无意义词）

   - 模型调优：

      - 尝试其他分类器（如SVM、随机森林）

      - 调整朴素贝叶斯的平滑参数（alpha）

#### 6. 依赖库

   - jieba：中文分词

   - scikit-learn：机器学习模型

   - numpy：数值计算

### 1. 高频词（Count-based）模式

实现代码
```commandline
from sklearn.feature_extraction.text import CountVectorizer

# 使用高频词 (词频统计)
vectorizer = CountVectorizer(tokenizer=get_words, max_features=100)  # 保留前100个高频词
X_train = vectorizer.fit_transform([" ".join(words) for words in all_words])
```

特点

直接统计每个词的出现次数

   - 简单快速，但忽略词的重要性差异

   - 适合短文本或词频本身具有强区分性的场景

### 2. TF-IDF模式

实现代码
```commandline
from sklearn.feature_extraction.text import TfidfVectorizer

# 使用TF-IDF特征
vectorizer = TfidfVectorizer(tokenizer=get_words, max_features=100)  # 保留前100个重要词
X_train = vectorizer.fit_transform([" ".join(words) for words in all_words])

```

特点

   - 计算词频（TF） × 逆文档频率（IDF）

   - 抑制高频常见词（如"的"、"是"），突出重要词

   - 适合长文本或需要区分词重要性的场景

### 3. 完整切换实现

#### （1）定义特征提取器

```commandline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_feature_extractor(mode='tfidf', top_n=100):
    """选择特征提取模式"""
    if mode == 'count':
        return CountVectorizer(
            tokenizer=get_words, 
            max_features=top_n
        )
    elif mode == 'tfidf':
        return TfidfVectorizer(
            tokenizer=get_words,
            max_features=top_n
        )
    else:
        raise ValueError("模式必须是 'count' 或 'tfidf'")
```

#### （2）修改训练流程

```commandline
# 选择模式（'count'或'tfidf'）
feature_mode = 'tfidf'  

# 获取特征提取器
vectorizer = get_feature_extractor(mode=feature_mode, top_n=100)

# 转换训练数据
X_train = vectorizer.fit_transform([" ".join(words) for words in all_words])
labels = np.array([1]*127 + [0]*24)

# 训练模型
model = MultinomialNB()
model.fit(X_train, labels)

# 预测函数修改
def predict(filename, vectorizer, model):
    words = get_words(filename)
    X = vectorizer.transform([" ".join(words)])
    result = model.predict(X)
    return '垃圾邮件' if result == 1 else '普通邮件'
```

### 4. 两种模式对比

|   特性   | 高频词模式 |TF-IDF模式 |
|:------:|:-------:|:---:|
| 计算方式 | 原始词频统计 |  词频 × 逆文档频率   |
| 对常见词的处理 | 保留所有高频词 |  降低常见词权重   |
|  适合场景  |   短文本、关键词直接匹配   |  长文本、需要区分词重要性   |
| 计算复杂度  |   低   |  略高   |
| sklearn类 |   CountVectorizer   |  TfidfVectorizer   |

### 5. 示例调用

```commandline
# 初始化
vectorizer = get_feature_extractor(mode='tfidf')  # 切换为TF-IDF模式
model = MultinomialNB()

# 训练
X_train = vectorizer.fit_transform([" ".join(words) for words in all_words])
model.fit(X_train, labels)

# 预测
print(predict('邮件_files/151.txt', vectorizer, model))
```

### 6. 关键注意事项

   - 分词一致性：

      - 确保get_words函数在训练和预测时处理方式一致

   - 特征维度：

      - max_features参数控制特征数量（建议100-500）

   - 停用词处理：

      - 可在CountVectorizer/TfidfVectorizer中添加stop_words参数：

       TfidfVectorizer(stop_words=['的', '是', '在'], ...)

![运行结果](https://github.com/sea-ka/ai.task4/tree/master/images/task.png)
