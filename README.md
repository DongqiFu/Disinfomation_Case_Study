# Disinformation Study

## Preliminary
* **Load Dataset**. There are two datasets and one pre-trained language model need to be downloaded and placed in the "fake-and-real-news-dataset" folder. They are (1) [fake news data](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv) (23,538 articles), (2) [real news data](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=True.csv) (21,418 articles), and (3) [Google pre-trained word2vec model](https://code.google.com/archive/p/word2vec/) (3 million words and each has a 300-dim vector). Also, you have the backup online addresses, fake news data [here](https://drive.google.com/file/d/1T798b0Qi4AB6GzOTccbsCaPmhSI_0iN9/view?usp=sharing), real news data [here](https://drive.google.com/file/d/15mOoPsUaI9OeWiHJ5XP-u_oDlrxzeo8z/view?usp=sharing), and pre-trained word2vec [here](https://drive.google.com/file/d/1W8EfxWRBchX_c6ShC6neZRKlokhPV4tR/view?usp=sharing).

* **Required Libraries**.
  - numpy 1.20.1
  - scipy 1.6.2
  - pandas 1.2.4
  - nltk 3.6.2
  - gensim 4.0.1
  - sklearn 0.24.1
  - torch 1.9.0

* **Article Embedding**
  - [Word Graph Construction] We contrust an undirected word graph for each news article. Briefly, if two words co-occur in a length-specified sliding window, then there will be an edge connecting these two words. For example, "I eat an apple" and the length of the window is 2, then edges could be {I-eat, I-an, eat-an, eat-apple} (with stop words kept). More details of constructing a word graph can be found at [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf).
  - [Geometric Feature Extraction] We use the idea of the [SDG model](https://github.com/DongqiFu/SDG) to obtain node embeddings. Briefy, a node's representation is aggregated based on its personalized PageRank vector weighted neighours' features. Then we call any pooling function (like sum pooling or mean pooling) to aggregate node embeddings to obtain the graph-level representation vector for each constructed word graph.
  - You can run "sdg_model.py" to extract vector representation for each news article, which will then store a feature matrix and a label matrix of all appeared fake/real news articles. Also, you can access the extracted [feature_matrix.pkl](https://drive.google.com/file/d/1TtAc6rBs5rxCyvqMqjWyCtsjWfpl7Mgn/view?usp=sharing) and [label_matrix.pkl](https://drive.google.com/file/d/1Drdyr0WiCbK6KV2TXYVSdMqPvJcK2Eni/view?usp=sharing), and put them into the root directory of this repository.

## Study Directions
* **1. Detection Effectivess**
  - 1.1 [Accuracy] Run "classification_acc.py", which is responsible for training an accuracy-acceptable classifier (e.g. MuFasa model by "mufasa.py" or MLP by sklearn) and saving the classification model
  - 1.2 [Precision, Recall, and F1-score] Run "other_metrics.py" to test the trained classider in terms of other metrics, like precision, recall, and F1-score.
* **2. Explanation**
  - 2.1 [Misleading Degree] Run "top_n_words.py" to find top n misleading words in a news article. For example, with word _w_ the fake article news a is detected as fake news with probability _p_, without word _w_ the fake article news a is detected as fake news with probability _q_, then the probability gain of _w_ to a is _(q-p)_.
* **3. Robustness**
  - 3.1 [Varying Feature Dimensions] Run "dimension_redution.py" to reduce the aricle embedding dimension by PCA, and train a new classifier (e.g. MuFasa model by "mufasa.py" or MLP by sklearn) on the truncated feature matrix.
  - 3.2 Label Noise Injection
