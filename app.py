import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Function to get mean pooling embeddings
def get_mean_pooling_embeddings(text_list, tokenizer, model, device, batch_size=1):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch_text = text_list[i:i+batch_size]
            batch_text = [str(text) for text in batch_text]
            encoded_input = tokenizer(batch_text, padding=True, truncation=True, return_tensors='pt', max_length=512)
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled_embeddings = sum_embeddings / sum_mask
            embeddings.append(mean_pooled_embeddings.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

# Load the data
@st.cache_data
def load_data():
    try:
        labs = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/AKmsNHd_-KnDvXWzNhaSrw/Labs.csv')
        selected_labs = labs[(labs['status'] == 'Published') & (labs['language'] =='English')].copy()
        selected_labs['description'] = selected_labs['name'] + ' ' + selected_labs['short_description'].fillna('')
        unique_labs = selected_labs[['id', 'description']].drop_duplicates().reset_index(drop=True)
        return unique_labs
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Generate and reduce embeddings
@st.cache_data
def process_embeddings(descriptions):
    tokenizer, model, device = load_model()
    if tokenizer is None:
        return None, None, None, None
    
    # Get embeddings
    embeddings = get_mean_pooling_embeddings(descriptions, tokenizer, model, device)
    
    # Scale embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings.numpy())

    # Use PCA with n_components=2 for visualization
    reducer_2d = PCA(n_components=2, random_state=42)
    embeddings_reduced_2d = reducer_2d.fit_transform(embeddings_scaled)
    embeddings_reduced_2d = normalize(embeddings_reduced_2d)

    # Use PCA with n_components=15 for clustering
    reducer_clustering = PCA(n_components=15, random_state=42)
    embeddings_reduced_clustering = reducer_clustering.fit_transform(embeddings_scaled)
    embeddings_reduced_clustering = normalize(embeddings_reduced_clustering)

    return embeddings_reduced_2d, embeddings_reduced_clustering, scaler, reducer_clustering

# Perform clustering
@st.cache_resource
def perform_clustering(embeddings_reduced_clustering, k=15):
    if embeddings_reduced_clustering is None:
        return None
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(embeddings_reduced_clustering)
    return kmeans

# Find recommendations
def find_recommendations(query, unique_labs, kmeans_model, embeddings_reduced_clustering, 
                        scaler, reducer_clustering, tokenizer, model, device, num_recommendations=5):
    try:
        # Get query embedding
        query_embedding = get_mean_pooling_embeddings([query], tokenizer, model, device)
        
        # Scale query embedding using the fitted scaler
        query_embedding_scaled = scaler.transform(query_embedding.numpy())
        
        # Transform query embedding using the fitted reducer
        query_embedding_reduced = reducer_clustering.transform(query_embedding_scaled)
        query_embedding_reduced = normalize(query_embedding_reduced)

        # Find the cluster of the query
        query_cluster = kmeans_model.predict(query_embedding_reduced)[0]

        # Filter labs in the same cluster
        cluster_mask = kmeans_model.labels_ == query_cluster
        relevant_labs = unique_labs[cluster_mask].copy()

        # Calculate similarity between query and relevant labs
        relevant_embeddings_reduced = embeddings_reduced_clustering[cluster_mask]
        similarities = np.dot(relevant_embeddings_reduced, query_embedding_reduced.T).flatten()

        # Add similarities to the relevant_labs DataFrame
        relevant_labs = relevant_labs.reset_index(drop=True)
        relevant_labs['similarity'] = similarities

        # Sort by similarity and return top recommendations
        recommendations = relevant_labs.sort_values(by='similarity', ascending=False).head(num_recommendations)

        return recommendations
    except Exception as e:
        st.error(f"Error finding recommendations: {e}")
        return pd.DataFrame()

# Streamlit App
def main():
    st.title("Course Recommendation System")
    st.markdown("Find courses similar to your interests using AI-powered recommendations!")
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        unique_labs = load_data()
        
        if unique_labs.empty:
            st.error("Failed to load course data.")
            return
            
        tokenizer, model, device = load_model()
        
        if tokenizer is None:
            st.error("Failed to load AI model.")
            return
        
        # Process embeddings
        embeddings_reduced_2d, embeddings_reduced_clustering, scaler, reducer_clustering = process_embeddings(
            unique_labs['description'].tolist()
        )
        
        if embeddings_reduced_clustering is None:
            st.error("Failed to process course embeddings.")
            return
            
        # Perform clustering
        kmeans_model = perform_clustering(embeddings_reduced_clustering)
        
        if kmeans_model is None:
            st.error("Failed to cluster courses.")
            return

    # Add cluster labels to the dataframe
    unique_labs['cluster'] = kmeans_model.labels_
    
    st.success(f"Loaded {len(unique_labs)} courses successfully!")

    # User input
    st.subheader("Find Your Perfect Course")
    query = st.text_input(
        "Enter a course description or topic:", 
        placeholder="e.g., Machine Learning, Data Science, Web Development"
    )

    # Get recommendations
    if st.button("Get Recommendations", type="primary"):
        if query.strip():
            with st.spinner("Finding the best courses for you..."):
                recommendations = find_recommendations(
                    query, unique_labs, kmeans_model, embeddings_reduced_clustering,
                    scaler, reducer_clustering, tokenizer, model, device
                )
                
            if not recommendations.empty:
                st.subheader("Recommended Courses:")
                for i, (index, row) in enumerate(recommendations.iterrows(), 1):
                    with st.expander(f"#{i} - Similarity: {row['similarity']:.3f}"):
                        st.write(row['description'])
                        st.write(f"**Course ID:** {row['id']}")
                        st.write(f"**Cluster:** {row['cluster']}")
            else:
                st.warning("No recommendations found. Try a different query.")
        else:
            st.warning("Please enter a query.")

    # Visualization option
    st.subheader("Data Visualization")
    if st.checkbox("Show Clustering Visualization (2D PCA)"):
        if embeddings_reduced_2d is not None:
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(
                embeddings_reduced_2d[:, 0], 
                embeddings_reduced_2d[:, 1], 
                c=unique_labs['cluster'], 
                cmap='tab20', 
                s=30,
                alpha=0.7
            )
            ax.set_title("Course Clustering Visualization (2D PCA)")
            ax.set_xlabel("PCA Dimension 1")
            ax.set_ylabel("PCA Dimension 2")
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label="Cluster")
            
            st.pyplot(fig)
            plt.close()  # Clean up memory
            
            # Show cluster statistics
            cluster_counts = unique_labs['cluster'].value_counts().sort_index()
            st.write("**Cluster Statistics:**")
            st.bar_chart(cluster_counts)

if __name__ == "__main__":
    main()