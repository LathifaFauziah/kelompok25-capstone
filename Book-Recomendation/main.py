# Setup
import streamlit as st
import pandas as pd
import hashlib

# Convert Pass into hash format
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

# Check password matches during login
def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# DB Management
import sqlite3 
conn = sqlite3.connect('user_data.db')
c = conn.cursor()
# DB Functions for create table
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,email TEX, password TEXT)')

# Insert the data into table
def add_userdata(username,email,password):
	c.execute('INSERT INTO userstable(username,email,password) VALUES (?,?,?)',(username,email,password))
	conn.commit()

# Password and email fetch
def login_user(email,password):
	c.execute('SELECT * FROM userstable WHERE email =? AND password = ?',(email,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data


# Mian function
def main():
	#"""Login page"""
	st.title("welcome! ")
	menu = ["Login","SignUp"]
	choice = st.selectbox("Select Login or SignUp from dropdown box ▾",menu,)
	st.markdown(
     "<h10 style='text-align: left; color: #ffffff;'> If you do not have an account, create an accouunt by select SignUp option from above dropdown box.</h10>",
     unsafe_allow_html=True
     )
	if choice == "":
		st.subheader("Login")
	elif choice == 'Login':
		st.write('-------')
		st.subheader('Log in to the App')

		email = st.text_input("User Name",placeholder='email')
		
		password = st.text_input("Password",type='password')
  
		if st.button("Login"):
			# if password == '12345':
			# Hash password creation and store in a table
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(email,check_hashes(password,hashed_pswd))
			if result:

				st.success("Logged In as {}".format(email))

				
				if st.success:
					st.subheader("User Profiles")
					user_result = view_all_users()
					clean_db = pd.DataFrame(user_result,columns=["Username","Email","Password"])
					st.dataframe(clean_db)
			else:
				st.warning("Incorrect Username/Password")
	elif choice == "SignUp":
		st.write('-----')
		st.subheader("Create New Account")
		new_user = st.text_input("Username",placeholder='name')
		new_user_email = st.text_input('Email id',placeholder='email')
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			if new_user == '':     # if user name empty then show the warnings
				st.warning('Inavlid user name')
			elif new_user_email == '':   # if email empty then show the warnings
				st.warning('Invalid email id')
			elif new_password == '':   # if password empty then show the warnings
				st.warning('Invalid password')
			else:
				create_usertable()
				add_userdata(new_user,new_user_email,make_hashes(new_password))
				st.success("You have successfully created a valid Account")
				st.info("Go up and Login to you account")



if _name_ == '_main_':
	main()
