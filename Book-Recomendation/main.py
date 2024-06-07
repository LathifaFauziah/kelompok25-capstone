import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Set the background colors
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0; /* Light gray background */
        margin: 0; /* Remove default margin for body */
        padding: 0; /* Remove default padding for body */
    }
    .st-bw {
        background-color: #eeeeee; /* White background for widgets */
    }
    .st-cq {
        background-color: #cccccc; /* Gray background for chat input */
        border-radius: 10px; /* Add rounded corners */
        padding: 8px 12px; /* Add padding for input text */
        color: black; /* Set text color */
    }
    .st-cx {
        background-color: white; /* White background for chat messages */
    }
    .sidebar .block-container {
        background-color: #f0f0f0; /* Light gray background for sidebar */
        border-radius: 10px; /* Add rounded corners */
        padding: 10px; /* Add some padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained KNN model for book recommendation
def load_knn_model():
    try:
        with open('book_recommendation_knn_model.pkl', 'rb') as file:
            model_knn = pickle.load(file)
        return model_knn
    except FileNotFoundError:
        st.error("Book Recommendation model file not found. Please ensure 'book_recommendation_knn_model.pkl' exists in the correct directory.")
        return None

model_knn = load_knn_model()

# Example data loading
def load_example_data():
    user_item_rating = pd.DataFrame({
        'user_id': [0, 1, 2, 3, 4],
        'book_1': [5, 4, 0, 0, 3],
        'book_2': [0, 5, 0, 4, 2],
        'book_3': [0, 0, 5, 0, 0],
        'book_4': [4, 0, 3, 0, 4],
        'book_5': [0, 2, 4, 5, 0]
    }).set_index('user_id')

    matrix = np.array([
        [5, 0, 0, 4, 0],
        [4, 5, 0, 0, 2],
        [0, 0, 5, 3, 4],
        [0, 4, 0, 0, 5],
        [3, 2, 0, 4, 0]
    ])

    return user_item_rating, matrix

# Function to recommend books based on user ID
def recommend_books(user_id, user_item_rating, matrix, num_recommendations=5):
    if model_knn is None:
        return ["Model not loaded."]

    # Check if user_id exists in user_item_rating
    if user_id not in user_item_rating.index:
        return ["User not found."]

    # Get the location of the user in the user_item_rating DataFrame
    user_loc = user_item_rating.index.get_loc(user_id)

    # Use the trained model to find nearest neighbors of the user
    distances, indices = model_knn.kneighbors(matrix[user_loc].reshape(1, -1), n_neighbors=num_recommendations + 1)
    similar_users = indices.flatten()[1:]  # Exclude the user itself

    # Dictionary to store recommended books with their cumulative ratings
    recommended_books = {}

    # Iterate over similar users and their ratings
    for index in similar_users:
        similar_user_ratings = user_item_rating.iloc[index]
        for book, rating in similar_user_ratings.items():
            if rating >= 4:  # Consider books rated 4 or higher by similar users
                if book in recommended_books:
                    recommended_books[book] += rating
                else:
                    recommended_books[book] = rating

    # Sort recommended books by cumulative rating and select top recommendations
    sorted_recommendations = sorted(recommended_books.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    return [book for book, _ in sorted_recommendations]

# Function to list available user IDs
def list_available_user_ids(user_item_rating):
    return user_item_rating.index.tolist()

# Load genre prediction model and fit the vectorizer
def load_genre_model():
    model_path = "book_recommendation_knn_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            genre_model = pickle.load(file)
        return genre_model
    else:
        st.error("Genre model file not found. Please ensure 'book_recommendation_knn_model.pkl' exists in the correct directory.")
        return None

loaded_genre_model = load_genre_model()

# Example text data to fit the vectorizer (replace with your actual data)
example_data = [
    "An adventurous journey through the mountains",
    "A thrilling crime story with unexpected twists",
    "A magical fantasy world with dragons and wizards",
    "Learning the basics of machine learning",
    "A heartwarming romance novel set in Paris",
    "A gripping thriller that will keep you on the edge of your seat"
]

# Fit the vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(example_data)

# Function to recommend books based on predicted genre
def recommend_books_by_genre(book_description):
    if loaded_genre_model is None:
        return ["No recommendations available due to missing genre model."]

    # Transform the book description into TF-IDF vectors
    X_new = vectorizer.transform([book_description])

    # Predict the genre of the book
    predicted_genre = loaded_genre_model.predict(X_new)[0]

    # Translate the predicted genre to corresponding genre name
    genre_mapping = {0: "adventure", 1: "crime", 2: "fantasy", 3: "learning", 4: "romance", 5: "thriller"}
    recommended_genre = genre_mapping[predicted_genre]

    # Example recommendation based on genre (replace with your actual recommendation logic)
    if recommended_genre == "adventure":
        recommended_books = ["The Hobbit", "Treasure Island", "Jurassic Park"]
    elif recommended_genre == "crime":
        recommended_books = ["The Girl with the Dragon Tattoo", "Gone Girl", "The Da Vinci Code"]
    elif recommended_genre == "fantasy":
        recommended_books = ["Harry Potter and the Philosopher's Stone", "The Name of the Wind", "Mistborn"]
    elif recommended_genre == "learning":
        recommended_books = ["Sapiens: A Brief History of Humankind", "Thinking, Fast and Slow", "Educated"]
    elif recommended_genre == "romance":
        recommended_books = ["Pride and Prejudice", "The Notebook", "Me Before You"]
    else:  # thriller
        recommended_books = ["The Silence of the Lambs", "The Girl on the Train", "The Bourne Identity"]

    return recommended_books

# Streamlit app
def main():
    st.title("Book Recommendation System")

    # Load example data
    user_item_rating, matrix = load_example_data()

    # Sidebar for book recommendation
    st.sidebar.header("User-based Book Recommendation System")
    st.sidebar.write("Available User IDs:")
    user_ids = list_available_user_ids(user_item_rating)
    st.sidebar.write(user_ids)

    user_id_input = st.sidebar.text_input("Enter User ID:")
    recommend_btn = st.sidebar.button("Recommend")

    if recommend_btn:
        try:
            user_id = int(user_id_input)
            if user_id in user_ids:
                st.write(f"Recommended books for User-ID: {user_id}")
                recommendations = recommend_books(user_id, user_item_rating, matrix)
                if isinstance(recommendations, list):
                    st.write("Recommended Books:")
                    for book in recommendations:
                        st.write(book)
                else:
                    st.write(recommendations)
            else:
                st.write("User ID not found. Please enter a valid User ID.")
        except ValueError:
            st.write("Invalid input! Please enter a numeric User ID.")

    st.header("Book Recommendation by Genre")
    # Input for book description
    book_description = st.text_input("User-ID Book Recommendation:")

    # Recommendation button
    if st.button("Get Recommendations"):
        if not book_description:
            st.warning("Please enter the description of the book.")
        else:
            recommended_books = recommend_books_by_genre(book_description)
            st.write("Recommended Books:")
            for book in recommended_books:
                st.write(book)

if __name__ == "__main__":
    main()
