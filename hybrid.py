from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from multi_hot_encoder import MultiHotEncoder

from gensim.models import Word2Vec

app = FastAPI()

# Load pre-trained models and datasets
preprocessor = joblib.load('model/column_transformer.pkl')
word_vec_model = Word2Vec.load('model/word2vec_model.model')
knn = joblib.load('model/knn_model.pkl')
products = pd.read_csv("data/final-products.csv")
user_product_data = pd.read_csv('data/user-product-data.csv')
svd_model = joblib.load('data/svd-model.pkl')
product_index = pd.read_csv('data/product-index.csv', index_col=0)
user_item_matrix = pd.read_csv('data/user-item-matrix.csv', index_col=0)
user_item_matrix.columns = user_item_matrix.columns.astype(int)

# Calculate Top 10 Products
product_scores = user_item_matrix.sum(axis=0)
top_products_ids = product_scores.sort_values(ascending=False).index.astype('int').tolist()
top_products = products[products['Product ID'].isin(top_products_ids)].set_index('Product ID')
top_products = top_products.reindex(top_products_ids).reset_index()

# Prepare product features including both numerical and textual data (from Word2Vec)
def get_word2vec_vector(text, model):
    tokens = text.lower().split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
def prepare_products(products, preprocessor, word_vec_model):
    # Preprocess non-text features
    products['Multi Categories'] = products['Categories'].str.split(',')
    products['Size'] = products['Size'].fillna('')

    product_features = products.drop(
        ['Product ID', 'Status', 'Stock Status', 'Categories', 'Name', 'Description', 'Short Description'],
        axis=1, errors='ignore'
    )
    product_features = preprocessor.transform(product_features)

    name_vectors = np.array([get_word2vec_vector(text, word_vec_model) for text in products['Name']])
    description_vectors = np.array([get_word2vec_vector(text, word_vec_model) for text in products['Description']])
    short_desc_vectors = np.array([get_word2vec_vector(text, word_vec_model) for text in products['Short Description']])
    text_features_combined = np.hstack([name_vectors, description_vectors, short_desc_vectors])

    product_vector = np.hstack([product_features, text_features_combined])
    return product_vector
    
product_vectors = prepare_products(products, preprocessor, word_vec_model)

# Function to get product recommendations based on KNN
def recommend_products(product_index, knn_model, X, products, n_recommendations=5):
    product_vector = X[product_index].reshape(1, -1)
    distances, indices = knn_model.kneighbors(product_vector, n_neighbors=n_recommendations + 1)
    similar_product_indices = indices.flatten()[1:]
    recommended_products = products.iloc[similar_product_indices].copy()
    recommended_products['Similarity'] = 1 - distances.flatten()[1:]
    return recommended_products

# Function to get content-based recommendations for a set of products
def recommend_content(user_product_indices, knn_model, X, products, n_recommendations=5):
    if not user_product_indices:
        # Fallback to top products if no user history
        top_products['Similarity'] = 1
        return top_products
    all_recommendations = pd.DataFrame()
    for product_index in user_product_indices:
        recommendations = recommend_products(product_index, knn_model, X, products, n_recommendations)
        all_recommendations = pd.concat([all_recommendations, recommendations])
    all_recommendations = all_recommendations.drop_duplicates(subset='Product ID')
    all_recommendations = all_recommendations.sort_values(by='Similarity', ascending=False)
    return all_recommendations

# Function to get collaborative recommendations based on user purchases and SVD model
def recommend_collab(user_id, user_purchases, svd_model, products, n_recommendations=5):
    if user_purchases.empty:
        # Fallback to top products if no user history
        top_products['Predicted Rating'] = 1
        return top_products
    all_recommendations = pd.DataFrame()
    user_matrix = user_purchases.values.reshape(1, -1)
    user_svd = svd_model.transform(user_matrix)
    user_recommendations = np.dot(user_svd, svd_model.components_)
    product_scores = user_recommendations.flatten()
    product_scores = pd.Series(product_scores)
    product_scores = product_scores[product_scores > 0]
    recommended_indices = product_scores.sort_values(ascending=False).index.tolist()
    all_recommendations = products.iloc[recommended_indices].copy()
    all_recommendations['Predicted Rating'] = product_scores[recommended_indices]
    return all_recommendations

class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    product_ids: Optional[List[int]] = None
    n_recommendations: Optional[int] = None

# API Endpoint: Recommend Product to User 
@app.post("/recommend/")
def recommend(request: RecommendationRequest):
    user_id = request.user_id
    product_ids = request.product_ids
    n_recommendations = request.n_recommendations
    content_weight=0.5
    collaborative_weight=0.5
    
    try:
        # Prepare the list of product IDs based on user purchase history and input products
        base_product_ids = []
        user_purchases = pd.Series(0, index=user_item_matrix.columns)
        if (user_id) & (user_id in user_item_matrix.index):
            user_purchases = user_item_matrix.loc[user_id]
            product_bought_ids = user_product_data[(user_product_data['User ID'] == user_id) & (user_product_data['Product Bought'] != 0)]['Product ID'].tolist()
            base_product_ids.extend(product_bought_ids)
        
        if product_ids:
             # Update user purchases with new product interaction
            new_products = [pid for pid in product_ids if pid not in base_product_ids]
            user_purchases.loc[product_ids] = 1
            base_product_ids.extend(new_products)

        # Get the product indices corresponding to the user-purchased products
        product_bought = products[products['Product ID'].isin(base_product_ids)]
        product_indices = product_bought.index.tolist()

        # Generate recommendations if products exist in base_product_ids
        if base_product_ids:
            collaborative_recommendations = recommend_collab(user_id, user_purchases, svd_model, products, n_recommendations)
    
            content_recommendations = recommend_content(product_indices, knn, product_vectors, products, n_recommendations)
        
            hybrid_recommendations = pd.merge(content_recommendations[['Product ID', 'Similarity']], collaborative_recommendations[['Product ID', 'Predicted Rating']], on='Product ID', how='outer')
            hybrid_recommendations = pd.merge(products, hybrid_recommendations, on='Product ID', how='outer')
            hybrid_recommendations['Similarity'] = hybrid_recommendations['Similarity'].fillna(0)
            hybrid_recommendations['Predicted Rating'] = hybrid_recommendations['Predicted Rating'].fillna(0)
            hybrid_recommendations['Hybrid Similarity'] = content_weight * hybrid_recommendations['Similarity'] + collaborative_weight * hybrid_recommendations['Predicted Rating']
        
            hybrid_recommendations = hybrid_recommendations.sort_values(by='Hybrid Similarity', ascending=False)
            hybrid_recommendations = hybrid_recommendations[hybrid_recommendations['Hybrid Similarity'] > 0]
            filtered_recommendations = hybrid_recommendations[(hybrid_recommendations['Stock Status'] == 1) & (hybrid_recommendations['Status'] == 'publish')][:n_recommendations]
        else:
             # Handle case for new user by recommending top products
            top_products['Hybrid Similarity'] = 1
            filtered_recommendations = top_products[(top_products['Stock Status'] == 1) & (top_products['Status'] == 'publish')][:n_recommendations]

        # Fallback: If not enough recommendations, add top products
        if len(filtered_recommendations) < n_recommendations:
            filtered_top_products = top_products[(top_products['Stock Status'] == 1) & (top_products['Status'] == 'publish')].head(n_recommendations - len(filtered_recommendations))
            filtered_recommendations = pd.concat([filtered_recommendations, top_products], ignore_index=True)
    
        # Return the final recommendations in the specified format
        recommendations = filtered_recommendations[['Product ID', 'Name', 'Hybrid Similarity']].to_dict(orient="records")
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API Endpoint: Related Products
@app.get("/related/")
def related_products(product_id: int, n_related: Optional[int] = 5):
    product_index = products[products['Product ID'] == product_id].index[0]
    recommendations = recommend_products(product_index, knn, product_vectors, products, n_recommendations=n_related)
    recommendations = recommendations[(recommendations['Stock Status'] == 1) & (recommendations['Status'] == 'publish')]
    related = recommendations[['Product ID', 'Name', 'Similarity']].to_dict(orient="records")
    return related

# API Endpoint: Best Seller
@app.get("/bestsellers/")
def best_sellers(n_top: int = 10):
    total_sales = user_item_matrix.sum(axis=0).sort_values(ascending=False)
    top_product_ids = total_sales.head(n_top).index.tolist()
    best_products = products[products['Product ID'].isin(top_product_ids)].copy()
    best_products = best_products[(best_products['Stock Status'] == 1) & (best_products['Status'] == 'publish')]
    best_sellers = best_products[['Product ID', 'Name']].to_dict(orient="records")
    return best_sellers

# Get frequently bought together products
@app.get("/frequently_bought_together/")
def frequently_bought_together(product_id: int, n_recommendations: int = 5):
    try:
        product_vector = svd_model.components_[:, user_item_matrix.columns.get_loc(product_id)].reshape(1, -1)
        product_scores = np.dot(product_vector, svd_model.components_).flatten()
        sorted_indices = product_scores.argsort()[::-1]
        recommended_product_ids = user_item_matrix.columns[sorted_indices]
        recommended_product_ids = [pid for pid in recommended_product_ids if pid != product_id][:n_recommendations]
        recommended_products = products[products['Product ID'].isin(recommended_product_ids)].copy()
        recommended_products['Predicted'] = recommended_products['Product ID'].map(
            lambda pid: product_scores[user_item_matrix.columns.get_loc(pid)]
        )
        recommended_products = recommended_products.sort_values(by="Predicted", ascending=False)
        recommended_products = recommended_products[(recommended_products['Stock Status'] == 1) & (recommended_products['Status'] == 'publish')]
        frequently_bought_together = recommended_products[['Product ID', 'Name', 'Predicted']].to_dict(orient="records")
        return frequently_bought_together

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health Check Endpoint
@app.get("/")
def read_root():
    return {"message": "Recommendation system is up and running!"}