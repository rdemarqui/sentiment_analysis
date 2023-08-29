# Sentiment Analysis
<p align="center">
<img src="images\sent_analysis_cover.png" class="center" width="60%"/>
</p>

## Overview
Sentiment analysis is a segment of machine learning that deciphers emotions within textual data. By employing sophisticated algorithms, it classifies text as positive, negative, or neutral, enabling invaluable insights across industries. From enhancing customer experiences to gauging public opinion, sentiment analysis shapes decision-making in our data-driven world.

There are many off-the-shelf solutions to work with text data, especially for the English language. Unfortunately, those options turn restricted when we need to work with the Portuguese language. To explore some possible solutions, in this study, a series of methodologies were applied, both in the data preprocessing and in the text embedding, as can be seen further in the methodology section

## Objectives
Measure the impact of different text pre-processing methodologies for portuguese language, such as stop word removal, lemmatization and stemming in different types of word embedding, from the simplest ones like bag of words to LLM like BERT.

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

As can be seen in the chart below, stemming was the action that most reduced the text vocabulary size:

<p align="center">
<img src="images\voc_size.png" class="center" width="50%"/>
</p>

After text preprocessing, several text vectorization (embedding) methods were tested in each of the six text columns. First, we used `sklearn` to implement Bag of Words and TF-IDF. Afterward, we used `gensim` to implement Word2Vec (CBOW and Skip-gram), FastText, and Doc2Vec (DBOW and DM). Finally, two Portuguese fine-tuned pre-trained models were implemented: BERT `neuralmind/bert-base-portuguese-cased`[2] and Sentence Transformer `rufimelo/bert-large-portuguese-cased-sts`[3], both available on the HuggingFace website.

Most embedding models give a vector for each word; in those cases, a mean was applied, resulting in a (1, n) vector.

For text classification, we chose lightgbm due to its good accuracy, robustness, and speed. This part of implementation is avaiable on [Vectorization](https://github.com/rdemarqui/sentiment_analysis/blob/main/02%20Vectorization.ipynb) notebook.

## Results and Conclusions

To check if this results is consistents this same code could be applied in other dataframes. On Github there are other good data as well [5][6]. 

## References
* [1] https://github.com/americanas-tech/b2w-reviews01
* [2] https://huggingface.co/neuralmind/bert-base-portuguese-cased
* [3] rufimelo/bert-large-portuguese-cased-sts
* [4] https://arxiv.org/pdf/2201.03382.pdf
* [5] https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets
* [6] https://github.com/larifeliciana/books-reviews-portuguese
* [7] https://forum.ailab.unb.br/t/datasets-em-portugues/251
