import streamlit as st
import pandas as pd
import hashlib

# Convert Pass into hash format
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Check password matches during login
def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB Management
import sqlite3 
conn = sqlite3.connect('user_data.db')
c = conn.cursor()

# DB Functions for create table, insert data, and fetch users
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,email TEXT, password TEXT)')

def add_userdata(username, email, password):
    c.execute('INSERT INTO userstable(username,email,password) VALUES (?,?,?)',(username,email,password))
    conn.commit()

def login_user(email, password):
    c.execute('SELECT * FROM userstable WHERE email =? AND password = ?',(email,password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

# Book Recommendation Mechanics
def book_recommendation():
    books_df = pd.read_csv("Books.csv")
    users_df = pd.read_csv("Users.csv")
    ratings_df = pd.read_csv("Ratings.csv")

    merged_df = pd.merge(pd.merge(users_df, ratings_df, on="User-ID"), books_df, on="ISBN")

    book_ratings = merged_df.groupby("ISBN").agg({"User-ID": "count", "Book-Rating": "mean", "Book-Title": "first", "Year-Of-Publication": "first", "Book-Author": "first"}).reset_index()
    sorted_ratings = book_ratings.sort_values(by=["User-ID", "Book-Rating", "ISBN"], ascending=False)
    recommended_books = sorted_ratings.head(10)
    return recommended_books

# Streamlit app
def main():
    st.title("Welcome to Book Recommendation App")
    st.header("Please Login")

    menu = ["Login", "SignUp"]
    choice = st.selectbox("Select Login or SignUp", menu)

    if choice == "Login":
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(email, check_hashes(password, hashed_pswd))
            if result:
                st.success("Logged In as {}".format(email))
                st.write('-----')
                st.subheader("User Profiles")
                user_result = view_all_users()
                clean_db = pd.DataFrame(user_result, columns=["Username", "Email", "Password"])
                st.dataframe(clean_db)

                st.subheader("Book Recommendations")
                recommendations = book_recommendation()
                st.write(recommendations[["Book-Title", "Book-Author", "Year-Of-Publication", "User-ID", "Book-Rating"]])
                
            else:
                st.warning("Incorrect Username/Password")

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_user_email = st.text_input("Email")
        new_password = st.text_input("Password", type="password")

        if st.button("Signup"):
            if new_user == "":
                st.warning("Invalid username")
            elif new_user_email == "":
                st.warning("Invalid email")
            elif new_password == "":
                st.warning("Invalid password")
            else:
                create_usertable()
                add_userdata(new_user, new_user_email, make_hashes(new_password))
                st.success("Successfully created an account. Please login.")

if __name__ == "__main__":
    main()
