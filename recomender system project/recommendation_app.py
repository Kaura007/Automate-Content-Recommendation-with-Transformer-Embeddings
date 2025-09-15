
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
import umap
import numpy as np

# Function to get mean pooling embeddings (copy from previous code)
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

# Load the data (copy from previous code)
@st.cache_data
def load_data():
    labs = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/AKmsNHd_-KnDvXWzNhaSrw/Labs.csv')
    selected_labs = labs[(labs['status'] == 'Published') & (labs['language'] =='English')].copy()
    selected_labs['description'] = selected_labs['name'] + ' ' + selected_labs['short_description'].fillna('')
    unique_labs = selected_labs[['id', 'description']].drop_duplicates().reset_index(drop=True)
    return unique_labs

# Load model and tokenizer (copy from previous code)
@st.cache_resource
def load_model():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, device

# Generate and reduce embeddings (copy from previous code, adapt for Streamlit caching)
@st.cache_data
def process_embeddings(descriptions, tokenizer, model, device):
    embeddings = get_mean_pooling_embeddings(descriptions, tokenizer, model, device)

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Use UMAP with n_components=2 for visualization
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
    embeddings_reduced_2d = reducer.fit_transform(embeddings_scaled)
    embeddings_reduced_2d = normalize(embeddings_reduced_2d)


    # Use UMAP with n_components=15 for clustering (based on elbow method)
    reducer_clustering = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=15, random_state=42)
    embeddings_reduced_clustering = reducer_clustering.fit_transform(embeddings_scaled)
    embeddings_reduced_clustering = normalize(embeddings_reduced_clustering)

    return embeddings_reduced_2d, embeddings_reduced_clustering

# Perform clustering (copy from previous code, adapt for Streamlit caching)
@st.cache_resource
def perform_clustering(embeddings_reduced_clustering, k=15):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(embeddings_reduced_clustering)
    return kmeans

# Find recommendations
def find_recommendations(query, unique_labs, kmeans_model, embeddings_reduced_clustering, tokenizer, model, device, num_recommendations=5):
    query_embedding = get_mean_pooling_embeddings([query], tokenizer, model, device)
    scaler = StandardScaler()
    # Fit scaler on the training data (embeddings_scaled from process_embeddings) and then transform the query
    # This requires storing or re-calculating the scaler from the training data, which is not ideal for
    # caching. A simpler approach for demonstration is to fit the scaler on the query embedding and the training embeddings
    # combined, or use the scaler fitted on the training data if available.
    # For simplicity in this demo, we will refit the scaler. In a real application, save and load the scaler.
    all_embeddings = np.vstack((embeddings.numpy(), query_embedding.numpy())) # assuming 'embeddings' is available globally or passed
    scaler.fit(all_embeddings)
    query_embedding_scaled = scaler.transform(query_embedding.numpy())


    # Reduce dimensionality of query embedding using the fitted reducer
    # Need to refit reducer as well, or save and load it. Refitting here for simplicity.
    reducer_clustering = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=15, random_state=42)
    all_embeddings_reduced = reducer_clustering.fit_transform(all_embeddings) # assuming 'embeddings_scaled' is available globally or passed
    query_embedding_reduced = normalize(all_embeddings_reduced[-1].reshape(1, -1))


    # Find the cluster of the query
    query_cluster = kmeans_model.predict(query_embedding_reduced)[0]

    # Filter labs in the same cluster
    relevant_labs = unique_labs[kmeans_model.labels_ == query_cluster].copy()

    # Calculate similarity between query and relevant labs using reduced embeddings
    # Need to ensure embeddings_reduced_clustering is available and corresponds to unique_labs
    relevant_embeddings_reduced = embeddings_reduced_clustering[kmeans_model.labels_ == query_cluster]
    similarities = np.dot(relevant_embeddings_reduced, query_embedding_reduced.T).flatten()

    # Add similarities to the relevant_labs DataFrame
    relevant_labs['similarity'] = similarities

    # Sort by similarity and return top recommendations
    recommendations = relevant_labs.sort_values(by='similarity', ascending=False).head(num_recommendations)

    return recommendations

# Streamlit App
st.title("Course Recommendation System")

unique_labs = load_data()
tokenizer, model, device = load_model()
embeddings_reduced_2d, embeddings_reduced_clustering = process_embeddings(unique_labs['description'].tolist(), tokenizer, model, device)
kmeans_model = perform_clustering(embeddings_reduced_clustering)

# Add cluster labels to the dataframe for display/analysis
unique_labs['cluster'] = kmeans_model.labels_

query = st.text_input("Enter a course description or topic:", "Machine Learning")

if st.button("Get Recommendations"):
    if query:
        recommendations = find_recommendations(query, unique_labs, kmeans_model, embeddings_reduced_clustering, tokenizer, model, device)
        st.subheader("Recommended Courses:")
        for index, row in recommendations.iterrows():
            st.write(f"- {row['description']}")
    else:
        st.warning("Please enter a query.")

# Optional: Display clustering results
if st.checkbox("Show Clustering Visualization (2D UMAP)"):
    st.subheader("Clustering Visualization (2D UMAP)")
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embeddings_reduced_2d[:, 0], embeddings_reduced_2d[:, 1], c=unique_labs['cluster'], cmap='viridis', s=10)
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    st.pyplot(fig)
