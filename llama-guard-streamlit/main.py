from model import moderate_with_template ,moderate_chat
import streamlit as st

st.title("Llama Guard Demo")

user_content = st.text_input("Enter your Input Prompt:")
model_response = st.text_input("Enter your Prompt Response:")

input_chat = [
    {"role": "user", "content": str(user_content)}
]

output_chat = [
    {"role": "user", "content": str(user_content)},
    {"role": "assistant", "content": str(model_response)},
]


#moderate_chat(output_chat)


if st.button("Evaluate Prompt"):
    if user_content:
        st.write("Generating response...")
        with st.spinner("Processing..."):
            st.write(moderate_with_template(input_chat))


if st.button("Evaluate Response"):
    if output_chat:
        st.write("Generating response...")
        with st.spinner("Processing..."):
            st.write(moderate_with_template(output_chat))