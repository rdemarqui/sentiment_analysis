<div align="center">
  <h1>Sentiment Analysis</h1>
</div>

<p align="center">
<img src="images\sent_analysis_cover.png" class="center" width="60%"/>
</p>

## Overview
Sentiment analysis is a segment of machine learning that deciphers emotions within textual data. By employing sophisticated algorithms, it classifies text as positive, negative, or neutral, enabling invaluable insights across industries. From enhancing customer experiences to gauging public opinion, sentiment analysis shapes decision-making in our data-driven world.

There are many off-the-shelf solutions to work with text data, especially for the English language. Unfortunately, those options turn restricted when we need to work with the Portuguese language. To explore some possible solutions, in this study, a series of methodologies were applied, both in the data preprocessing and in the text embedding, as can be seen further in the methodology section

## Objectives
Measure the impact of different text pre-processing methodologies for portuguese language, such as stop word removal, lemmatization and stemming in different types of word embedding, from the simplest ones like bag of words to transfomers like BERT.

## Tecnologies Used
* `python 3.9.16`
* `pandas 1.5.3`
* `numpy 1.23.5`
* `sklearn 1.2.2`
* `lightgbm 4.0.0`
* `matplotlib 3.7.1`
* `seaborn 0.12.2`
* `gensim 4.3.1`
* `torch 2.0.1+cu118`
* `transformers 4.32.1`
* `sentence_transformers 2.2.2`
 
## About the Data
For this study, we used the dataset **B2W-Reviews01** that is an open corpus of product reviews. It contains more than 130k e-commerce customer reviews, collected from the Americanas.com website between January and May, 2018. [1]

## Methodology
This project was divided into two stages: Preprocessing and Vectorization.

For pre-processing, we apply several text cleaning methodologies. We started by applying text cleaning, such as uncasing all words, removing punctuation, accentuations, and special characters. After that, we applied different methodologies for text normalization. For stemming, we used the `nltk` package, while for lemmatization, we used `spacy`. Finally, we combined some of these solutions, resulting in six columns: review_text_clean (uncased, punctuation, accentuation, and special characters removed); review_text_clean_stop (review_text_clean with stop words removed); review_text_clean_stem (review_text_clean stemmed); review_text_clean_stop_stem (review_text_clean with stop words removed and stemmed); review_text_clean_lemma (review_text_clean lemmatized); review_text_clean_stop_lemma (review_text_clean with stop words removed and lemmatized).

This process can be reproduced using the [Text Preprocessing](https://github.com/rdemarqui/sentiment_analysis/blob/main/01%20Text%20Preprocessing.ipynb) notebook.

As can be seen in chart 1, stemming was the action that most reduced the text vocabulary size:


<p align="center">
<img src="images\voc_size.png" class="center" width="50%"/>
</p>
<p align="center"><em>Chart 1 - Vocabulary size</em></p>


After text preprocessing, several text vectorization (embedding) methods were tested in each of the six text columns. First, we used `sklearn` to implement Bag of Words and TF-IDF. Afterward, we used `gensim` to implement Word2Vec (CBOW and Skip-gram), FastText, and Doc2Vec (DBOW and DM). Finally, two Portuguese fine-tuned pre-trained models were implemented: BERT `neuralmind/bert-base-portuguese-cased`[2] and Sentence Transformer `rufimelo/bert-large-portuguese-cased-sts`[3], both available on the HuggingFace website.

Most embedding models give a vector for each word; in those cases, a mean was applied, resulting in a (1, n) vector.

For text classification, we chose lightgbm due to its good accuracy, robustness, and speed. This part of implementation is avaiable on [Vectorization](https://github.com/rdemarqui/sentiment_analysis/blob/main/02%20Vectorization.ipynb) notebook.

## Results and Conclusions
In this topic, we will compare all results based on the test dataset.

In the table 1, we can check the score of each model applied in each text preprocessing method:

<p align="center">
<img src="images\overall_score.png" class="center" width="90%"/>
</p>
<p align="center"><em>Table 1 - Overall Score</em></p>

Comparing all preprocessing methods, on average, lemmatization brought the best result (review_text_clean_lemma = ROC 0.97125). Surprisingly, removing stop words did more harm than good in all cases.

The next chart compares all vectorization methods:

<p align="center">
<img src="images\vector_model_boxplot.png" class="center" width="60%"/>
</p>
<p align="center"><em>Chart 2 - Vectorization methods comparison</em></p>

Bag of Words, TF-IDF, and Word2Vec models showed similar results. FastText performed a little worse but gave very concise results. Apparently, it is indifferent to text preprocessing methods. Surprisingly, Doc2Vec performed worse than the others. Finally, BERT sentence transformer obtained the best result, but with wide variations among the preprocessing methods.

In the chart 3, we ranked the top 10 best results. We can see that BERT sentence transformer gave the first two best results. In third place, TF-IDF with stemming brought good results. Comparing the first place with the tenth, we see less than one point of difference, i.e., from 0.98493 to 0.97689.

<p align="center">
<img src="images\top10.png" class="center" width="80%"/>
</p>
<p align="center"><em>Chart 3 - Top 10 best aproach</em></p>

Finally, in the chart 4, we can compare the results of each of the models applied to each preprocessing method:

<p align="center">
<img src="images\score_comparison.png" class="center" width="80%"/>
</p>
<p align="center"><em>Chart 4 - Score comparison</em></p>

As noted earlier, BERT sentence transformer got the best result. Its best performance was with just clean text (review_text_clean), without any additional preprocessing.

An item that must be taken into account is the processing time, and in this case, BERT performed worse (even using GPU) when compared to the other models. To process the six different type of text, BERT spent 02:05:43 and obtained a maximum ROC of 0.984702, while TF-IDF spent 00:10:06 to process the same amount of data and obtained a maximum ROC of 0.97907. This trade-off needs to be pondered when implementing it in production.

**Future Improvements**
Despite relatively worse performances between models, the results obtained were very good. Frederico Souza and João Filho obtained good results too, using TF-IDF and Logistic Regression[4]. This is probably due to the quantity and quality of the available data. A way to check if these results are consistent is to use this same code applied to other dataframes [5][6][7].

## References
* [1] https://github.com/americanas-tech/b2w-reviews01
* [2] https://huggingface.co/neuralmind/bert-base-portuguese-cased
* [3] https://huggingface.co/rufimelo/bert-large-portuguese-cased-sts
* [4] https://arxiv.org/pdf/2201.03382.pdf
* [5] https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets
* [6] https://github.com/larifeliciana/books-reviews-portuguese
* [7] https://forum.ailab.unb.br/t/datasets-em-portugues/251
