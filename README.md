# Deep Learning to Solve TSP (Traveling Salesman Problem)

The main goal of the model, is to predict the shortest path between two points in constant time.

This thing will help us to find the shortest path between two nodes in a graph with linear time, by using intelligent search algorthimgs (i.e. hill climbing algorithm) with predications that have been made by the model as a heurtistic values.

## How does it work?

The Model depends on two main white papers:

1. [Shortest path distance approximation using deep learning techniques](https://arxiv.org/pdf/2002.05257)

This paper explains how neural networks can measure correlations between two nodes in a graph, the main focus on this paper is on social networks, Although; we have used that same algorithm with different type of graphs.

> Social media networks will be represented as unweighted, undirected graphs, unlike routes, that will be represented as directed, weighted graphs.

To solve this problem, we did some modifications on preprocessing algorithm that we were using which was __(node2vec)__

2.[node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)

Node2vec Algorithm is derived from the natural language processing algorithm __(word2vec)__,the main goal of this algorithm is changing the way of representing graphs.
The default graphs structure is not suitable for computers, so this algorithm can map nodes into points in a ___n-dimensional___ space, so the whole graph will be treated as matrix.

This algorithm is used as a preprocessing step before applying the above algorithm, and predict the distance between two nodes.

## Results that we get

actually we don't have that huge computing power, so we can use a large amount of data to training this model. for this reason we get some size-limited datasets to train the model on it and that's what we get

* [chesapeake](https://networkrepository.com/road-chesapeake.php) (nodes= 39, edges= 170

![accuracy histogram](https://github.com/MohammadAsDev/mobiospace/blob/main/final_results/chesapeake_results/200_100_50_nn/default_1.png)

* [euro-road](https://networkrepository.com/road-euroroad.php) (nodes= 1K, edges= 1K)

![accuracy histogram for euro-road](https://github.com/MohammadAsDev/mobiospace/blob/main/final_results/euroroad_results/final_1.png)
