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

preprocessor = joblib.load('model/column_transformer.pkl')
word_vec_model = Word2Vec.load('model/word2vec_model.model')
knn = joblib.load('model/knn_model.pkl')

products = pd.read_csv("data/final-products.csv")
user_item_matrix = pd.read_csv('data/user-item-matrix.csv', index_col=0)
user_product_data = pd.read_csv('data/user-product-data.csv')

# Calculate Top 10 Products
product_scores = user_item_matrix.sum(axis=0)
top_products_ids = product_scores.sort_values(ascending=False).index.astype('int').tolist()
top_products = products[products['Product ID'].isin(top_products_ids)].set_index('Product ID')
top_products = top_products.reindex(top_products_ids).reset_index()


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

def recommend_products(product_index, knn_model, X, products, n_recommendations=5):
    product_vector = X[product_index].reshape(1, -1)
    distances, indices = knn_model.kneighbors(product_vector, n_neighbors=n_recommendations + 1)
    similar_product_indices = indices.flatten()[1:]
    recommended_products = products.iloc[similar_product_indices].copy()
    recommended_products['Similarity'] = 1 - distances.flatten()[1:]
    return recommended_products
    
def recommend_for_user(user_product_indices, knn_model, X, products, n_recommendations=5):
    if not user_product_indices:
        top_products['Similarity'] = 1
        return top_products
    all_recommendations = pd.DataFrame()
    for product_index in user_product_indices:
        recommendations = recommend_products(product_index, knn_model, X, products, n_recommendations)
        all_recommendations = pd.concat([all_recommendations, recommendations])
    all_recommendations = all_recommendations.drop_duplicates(subset='Product ID')
    all_recommendations = all_recommendations.sort_values(by='Similarity', ascending=False)
    return all_recommendations

class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    product_ids: Optional[List[int]] = None
    n_recommendations: Optional[int] = None

# API Endpoints
@app.post("/recommend/")
def recommend(request: RecommendationRequest):
    user_id = request.user_id
    product_ids = request.product_ids
    n_recommendations = request.n_recommendations
    try:
        base_product_ids = []
        if user_id:
            product_bought_ids = user_product_data[(user_product_data['User ID'] == user_id) & (user_product_data['Product Bought'] != 0)]['Product ID'].tolist()
            base_product_ids.extend(product_bought_ids)
            if product_ids:
                new_products = [pid for pid in product_ids if pid not in product_bought_ids]
                base_product_ids.extend(new_products)
        elif product_ids:
            base_product_ids.extend(product_ids)
        
        product_bought = products[products['Product ID'].isin(base_product_ids)]
        product_indices = product_bought.index.tolist()
        user_recommendations = recommend_for_user(product_indices, knn, product_vectors, products, n_recommendations)
        filtered_recommendations = user_recommendations[(user_recommendations['Stock Status'] == 1) & (user_recommendations['Status'] == 'publish')][:n_recommendations]
        recommendations = filtered_recommendations[['Product ID', 'Name', 'Similarity']].to_dict(orient="records")
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoint
@app.get("/")
def read_root():
    return {"message": "Recommendation system is up and running!"}
