from model import moderate_with_template ,moderate_chat
import streamlit as st
import pandas as pd

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
            
data = {
    "Number": [1, 2, 3, 4, 5, 6],
    "Harm Type": ["Violence & Hate", "Sexual Content", "Guns & Illegal weapons", "Regulated or Controlled substances", "Suicide & Self Harm", "Criminal Planning"],
    "Examples": [
        "Race, color, religion, national origin, sexual orientation, gender, gender identity, or disability",
        "Erotic content",
        "Explosives, biological agents, or chemical weapons",
        "Illegal drugs, tobacco, alcohol, or cannabis",
        "Providing instructions or information on methods of self-harm",
        "Statements that encourage violence (considered violating under Violence)"
    ]
}

# Create a DataFrame without the default index
df = pd.DataFrame(data)

# Display the DataFrame using st.dataframe()
st.dataframe(df, index=False)