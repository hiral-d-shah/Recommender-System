from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import joblib
import numpy as np
import pandas as pd

app = FastAPI()

svd_model = joblib.load('data/svd-model.pkl')
user_item_matrix = pd.read_csv('data/user-item-matrix.csv', index_col=0)
product_index = pd.read_csv('data/product-index.csv', index_col=0)
user_item_matrix.columns = product_index.index
product_scores = user_item_matrix.sum(axis=0)
top_products = product_scores.sort_values(ascending=False).index.tolist()

class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 10

def get_recommendations(user_id, n_recommendations=3):
    if user_id not in user_item_matrix.index:
        recommended_products = [index_to_product[idx] for idx in top_products[:n_recommendations]]
        return recommended_products
        
    user_purchases = user_item_matrix.loc[user_id]
    user_matrix = user_purchases.values.reshape(1, -1)
    user_svd = svd_model.transform(user_matrix)
    user_recommendations = np.dot(user_svd, svd_model.components_)

    product_scores = user_recommendations.flatten()
    product_scores = pd.Series(product_scores)
    # Remove already purchased products
    product_scores = product_scores * (user_purchases.values == 0)
    # Remove products with score = 0 (not purchased by any of the neighbors)
    product_scores = product_scores[product_scores > 0]
    
    recommended_indices = product_scores.sort_values(ascending=False)[:n_recommendations].index.tolist()

    if len(recommended_indices) < n_recommendations:
        purchased_products = user_purchases[user_purchases > 0].index.tolist()
        remaining_count = n_recommendations - len(recommended_indices)
        purchased_products = [p for p in purchased_products if p not in recommended_indices]
        recommended_indices.extend(purchased_products[:remaining_count])
    # Add top products (most purchased products from all users)
    if len(recommended_indices) < n_recommendations:
        remaining_count = n_recommendations - len(recommended_indices)
        top_products_filtered = [p for p in top_products if p not in recommended_indices]
        recommended_indices.extend(top_products_filtered[:remaining_count])
    
    recommended_products = [index_to_product[idx] for idx in recommended_indices]
    return recommended_products

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    recommendations = get_recommendations(request.user_id, request.n_recommendations)
    recommended_products = pd.DataFrame(recommendations)
    return {"user_id": request.user_id, "recommended_products": recommended_products.to_dict()}
