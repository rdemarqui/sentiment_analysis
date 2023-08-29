# Sentiment Analysis
<p align="center">
<img src="images\sent_analysis_cover.png" class="center" width="60%"/>
</p>

## Overview
Sentiment analysis is a segment of machine learning that deciphers emotions within textual data. By employing sophisticated algorithms, it classifies text as positive, negative, or neutral, enabling invaluable insights across industries. From enhancing customer experiences to gauging public opinion, sentiment analysis shapes decision-making in our data-driven world.

There are manny on the shelf solutions to work with text data, specialty for english language. Unfortunately, those options turns resctricted when we need to work with portuguese language. To verify some possible solutions, in this study, a serie of methodologies were applied, both in the data preprocessing and in the text embedding, as can be seen further in methodology topic.

## Objectives
Measure the impact of different text pre-processing methodologies for portuguese language, such as stop word removal, lemmatization and stemming in different types of word embedding, from the simplest ones like bag of words to LLM like BERT. For the embedding models, the computational cost was also measured and will be taken into account.

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
This project was divided in two stages, divided in two notebooks.

Firstly we applyed several methodologies for text cleanning. We started applying text cleaning as uncasing all words, removing punctuation, accentuations and special characteres. After that, we applied different methodologies for text normalization. For stematization we used `nltk` package while in lemmatization wue used `spacy`. Finnaly we combined some of this solution having as final solution six columns: review_text_clean (uncased, punctuation, accentuation and special characters removed); review_text_clean_stop (review_text_clean with stop word removed); review_text_clean_stem (review_text_clean stematized); review_text_clean_stop_stem (review_text_clean, stop words removed and stematized); review_text_clean_lemma (review_text_clean lemmatized); review_text_clean_stop_lemma (review_text_clean, stop words removed and lemmatized).

As can be seen in the chart below, stematization was the action that more reduced the text vocabulary size:
<p align="center">
<img src="images\voc_size.png" class="center" width="40%"/>
</p>


## Results and Conclusions

## References
* [1] https://github.com/americanas-tech/b2w-reviews01
* 
* https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets
* https://arxiv.org/pdf/2201.03382.pdf
* https://github.com/fredericods/PTBR-Text-Classification

  
* https://sol.sbc.org.br/index.php/stil/article/view/17785
* https://forum.ailab.unb.br/t/datasets-em-portugues/251
* https://github.com/americanas-tech/b2w-reviews01
* https://github.com/larifeliciana/books-reviews-portuguese
