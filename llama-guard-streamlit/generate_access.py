import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ["bala", "arun","kannan"]
usernames = ["bala", "arun","kannan"]
passwords = ["bala123", "arun123","kannan123"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)