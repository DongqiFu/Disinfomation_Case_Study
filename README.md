# Disinformation Study

## Preliminary
* **Load Dataset**. There are three datasets need to be downloaded and placed in the "fake-and-real-news-dataset" folder. They are (1) fake news data (23,538 articles), (2) real news data (21,418 articles), and (3) [Google pre-trained word2vec model](https://code.google.com/archive/p/word2vec/) (3 million words and each has a 300-dim vector). Due to the Github storage limit, you can access fake news data [here](https://drive.google.com/file/d/1T798b0Qi4AB6GzOTccbsCaPmhSI_0iN9/view?usp=sharing), real news data [here](https://drive.google.com/file/d/15mOoPsUaI9OeWiHJ5XP-u_oDlrxzeo8z/view?usp=sharing), and pre-trained word2vec [here](https://drive.google.com/file/d/1W8EfxWRBchX_c6ShC6neZRKlokhPV4tR/view?usp=sharing).
* **Required Libraries**.
  - numpy 1.20.1
  - scipy 1.6.2
  - pandas 1.2.4
  - nltk 3.6.2
  - gensim 4.0.1
  - sklearn 0.24.1
  - torch 1.9.0

* **Methodology**
This repositpory stores the program for extracring features of fake/real news articles. The extraction consists of two stages.
- First, contrust an undirected word graph for each news article. Briefly, if two words co-occur in a length-specified sliding window, then there will be an edge connecting these two words. For example, "I eat an apple" and the length of the window is 2, then edges could be {I-eat, I-an, eat-an, eat-apple} (with stop words kept). More details of constructing a word graph can be found at [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf).
- Second, use the idea of the [SDG model](https://github.com/DongqiFu/SDG) to obtain node embeddings, and then call any pooling function (like sum pooling or mean pooling) to aggregate node embeddings to obtain the graph-level representation vector for each constructed word graph.

## Study Directions
* **1. Detection Effectivess**
  - 1.1 Accuracy
  - 1.2 Precision, Recall, and F1-score
* **2. Explanation**
* **3. Robustness**
  - 3.1 Varying Feature Dimensions
  - 3.2 Label Noise Injection




* **Run main.py**. Just run the main.py, it will load datasets, pre-trained model, and utils.py to get graph embeddings for each news article. The output will be saved as feature_matrix.pkl and label_matrix.pkl in the main directory. Due to the Github storage limit, you can access computed [feature_matrix.pkl](https://drive.google.com/file/d/1TtAc6rBs5rxCyvqMqjWyCtsjWfpl7Mgn/view?usp=sharing) and [label_matrix.pkl](https://drive.google.com/file/d/1Drdyr0WiCbK6KV2TXYVSdMqPvJcK2Eni/view?usp=sharing).

* First, put the [fake news data](https://drive.google.com/file/d/1T798b0Qi4AB6GzOTccbsCaPmhSI_0iN9/view?usp=sharing), [real news data](https://drive.google.com/file/d/15mOoPsUaI9OeWiHJ5XP-u_oDlrxzeo8z/view?usp=sharing), and [pre-trained word2vec](https://drive.google.com/file/d/1W8EfxWRBchX_c6ShC6neZRKlokhPV4tR/view?usp=sharing) into the "fake-and-real-news-dataset" folder
* Second, run "SDG_model.py" to extract vector representation for each news article, which will store a feature matrix and a label matrix
* Third, run "classification_acc.py", which is responsible for training an accuracy-acceptable classifier (e.g. MuFasa model by "mufasa.py" or MLP by sklearn) and saving the classification model
* Fourth, run "top_n_words.py" to find top n misleading words in a news article. For example, with word w the fake article news a is detected as fake news with probability p, without word w the fake article news a is detected as fake news with probability q, then the probability gain of w to a is (q-p).
* Fifth, run "dimension_redution.py" to reduce the feature dimension by PCA, and train a new classifier (e.g. MuFasa model by "mufasa.py" or MLP by sklearn) on the truncated feature matrix.
* Sixth, run "other_metrics.py" to test the trained classider in terms of other metrics, like precision, recall, and F1-score.
