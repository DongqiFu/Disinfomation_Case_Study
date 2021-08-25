# Disinformation Feature Extration

This repositpory stores the program for extracring features of fake/real news articles. The extraction consists of two stages.
- First, contrust an undirected word graph for each news article. Briefy, if two words co-occur in a length-specificed sliding window, then there will be an edge connecting these two words. For example, _"I eat an apple"_ and the length of the window is 2, then edges could be _{I-eat, I-an, eat-an, eat-apple}_ (with stop word kept). More details of constructing a word graph can be cound at [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf).
- Second, use the idea of the [SDG model](https://github.com/DongqiFu/SDG) to obtain node embeddings, and then call any pooling function (like sum pooling or mean pooling) to aggregate node embeddings to obtain the graph-level representation vector for each constructed word graph.

## To run the program

## Requirements
