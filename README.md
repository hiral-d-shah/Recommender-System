# Recommender System
This repository contains a personalized recommender system designed for WordPress WooCommerce e-commerce sites. By leveraging customer order history, the system generates tailored product recommendations to enhance user experience and boost sales. It is optimized for datasets with 40+ products and 10,000+ orders, using collaborative filtering to align suggestions with individual customer preferences.

## File Structure

### `Data Fetch.ipynb`
This notebook handles data extraction from MySQL databases, exporting the data as CSV files for preprocessing.

### `Data Cleaning Visualization Feature Engineering.ipynb`
Responsible for cleaning the raw data, creating visualizations, and engineering features required for training the recommendation model.

### `Collaborative Filtering Based Model.ipynb`
Contains the implementation of the collaborative filtering recommendation algorithm.

### `Collab.py`
A FastAPI-based live API that serves recommendations. It takes a user ID and the number of desired recommendations as input and returns a list of recommended product IDs.

### `Content Based Model.ipynb`
This notebook implements a content-based recommendation approach, using product metadata (Name, Description, Category, Price, etc) to suggest products similar to those previously interacted with by the user.

### `Content.py`
A FastAPI-based live API that serves content-based recommendations using K-Nearest Neighbors (KNN) and Word2Vec embeddings. It takes a user ID, product IDs, and the number of desired recommendations as input and returns a list of recommended product IDs.

### `Hybrid.py`
A FastAPI-based live API combining collaborative filtering and content-based recommendations for a hybrid approach.
It provides:
- Personalized Recommendations: Merges content-based and collaborative filtering outputs to deliver tailored suggestions.
- Best Sellers: Recommends top-performing products based on purchase frequency.
- Related Products: Suggests products similar to a specified product ID using content-based techniques.
- Frequently Bought Together: Identifies products often purchased together with a specified product ID using collaborative filtering and SVD latent factors.
