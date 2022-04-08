# WIKIPEDIA INFORMATION RETRIEVAL (IR) ENGINE

In this project, we created an IR engine for retrieve the best wiki-results for a given query.

We used common Python libraries, like Numpy and Pandas, Nltk (for text preprocess), and we saved our data (inverted index pickle files, page rank calculations, page views and more) in Coogle Cloud Platform (CGP).

We also used GCP to run our server (which created by using flask library), and published URL address to enables remote-users to query our engine and recieve results.

Code Structure:

* inverted_index_gcp.py - Used to create an inverted index object, and to create objects that are necessary for writing and reading the relevant files to the appropriate path in GCP.
* inverted_index_to_gcp.ipynb - Used to create wiki-document corpus, and to build three inverted index objects (one for the documents body, one for title and one for anchor text), and write them to the appropriate path in GCP, using inverted_index_gcp.py.
* queries_train.json - Given training set of queries, with the optimal retrieval results. Used for training and testing our retrieval engine.
* search_frontend.py - Used to create the server (flask), receive queries from clients and provide answers, using 6 - different search functions (use rank_functions.py functions).
* rank_functions.py - Contains implementations of various search functions that retrieve documents using various similarity metrics.
* BM_25_from_index.py - Contains implementations of a class that we use to retrieve using the BM25 similarity index.
* Engine_performance_measurments.ipynb - Contains implementations for calculating the quality of the engine results, according to the MAP@k index, as well as functions for generating graphs that show the quality of the engine results and the retrieval times for various engine versions.
