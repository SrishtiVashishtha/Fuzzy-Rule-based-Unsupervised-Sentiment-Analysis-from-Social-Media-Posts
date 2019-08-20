# Fuzzy-Rule-based-Unsupervised-Sentiment-Analysis-from-Social-Media-Posts

**Code for the paper**

[Fuzzy rule based unsupervised sentiment analysis from social media posts.](https://www.sciencedirect.com/science/article/pii/S0957417419305366)

**Description**

This paper performs the sentiment analysis of social media posts particularly tweets.In this paper, we compute the sentiment of social media posts using a novel set of fuzzy rules involving multiple lexicons and datasets. We have proposed a novel unsupervised nine fuzzy rule-based system that classifies a social media post into: positive, negative or neutral sentiment class. We perform a comparative analysis of our method on nine public twitter datasets, three sentiment lexicons, four state-of-the-art approaches for unsupervised Sentiment Analysis and one state-of-the-art method for supervised machine learning. Our results can give an insight to researchers to choose which lexicon is best for social media. The fusion of fuzzy logic with lexicons for sentiment classification provides a new paradigm in Sentiment Analysis. Our method can be adapted to any lexicon and any dataset (two-class or three-class sentiment). 

**Dataset**

Any twitter dataset can be used for this fuzzy rule system. The dataset should contain a column for storing text and another column for storing sentiment labels of text. Example: [Sentiment140](http://help.sentiment140.com/for-students), [SemEval 2017](http://alt.qcri.org/semeval2017/task4/), [SemEval 2016](http://alt.qcri.org/semeval2016/task4/), etc. A popular website for datasets is: [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php?format=&task=cla&att=&area=&numAtt=&numIns=greater1000&type=&sort=nameUp&view=table)

**Lexicon**

We have used three sentiment lexicons: SentiWordNet, AFINN and VADER.

**Running the model:**

vader_fuzzyrules.py contains the code for implementing the fuzzy rule system using VADER lexicon.

**Citation**

If using this code, please cite our work using :

>Vashishtha, Srishti, and Seba Susan. "Fuzzy Rule based Unsupervised Sentiment Analysis from Social Media Posts." Expert Systems with Applications (2019): 112834.
