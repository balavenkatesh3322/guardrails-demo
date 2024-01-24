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

# st.markdown("# Zero-shot technique")
with st.expander("Llama Guard Template", expanded=False):
    if st.button("Scan Input Prompt",key="llama_guard_template_input"):
        if user_content:
            st.write("Generating response...")
            with st.spinner("Processing..."):
                #st.write()
                model_output = moderate_with_template(input_chat)
                st.markdown(f"<div style='background-color: #ffff00;'>{model_output}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please provide a input Prompt.") 


    if st.button("Scan Model Response",key="llama_guard_template_output"):
        if user_content and model_response:
            st.write("Generating response...")
            with st.spinner("Processing..."):
                model_output = moderate_with_template(output_chat)
                st.markdown(f"<div style='background-color: #ffff00;'>{model_output}</div>", unsafe_allow_html=True)
                #st.write()
        else:
            st.warning("Please provide a Input and output Prompt.") 
            
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

    df = pd.DataFrame(data)
    st.markdown("Llama Guard Taxonomy")
    st.table(df)
    
with st.expander("Fine Tuned Template", expanded=False):
    if st.button("Scan Input Prompt",key="llama_guard_fine_tune_template_input"):
        if user_content:
            st.write("Generating response...")
            with st.spinner("Processing..."):
                #st.write()
                model_output = moderate_chat(input_chat)
                st.markdown(f"<div style='background-color: #ffff00;'>{model_output}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please provide a input Prompt.") 


    if st.button("Scan Model Response",key="llama_guard_fine_tune_template_output"):
        if user_content and model_response:
            st.write("Generating response...")
            with st.spinner("Processing..."):
                model_output = moderate_chat(output_chat)
                st.markdown(f"<div style='background-color: #ffff00;'>{model_output}</div>", unsafe_allow_html=True)
                #st.write()
        else:
            st.warning("Please provide a Input and output Prompt.") 
            
    data = {
        "Number": [1, 2, 3, 4, 5],
        "Harm Type": ["Violence & Hate", "Guns & Illegal weapons", "Regulated or Controlled substances", "Suicide & Self Harm", "Criminal Planning"],
        "Examples": [
            "Race, color, religion, national origin, sexual orientation, gender, gender identity, or disability",
            "Explosives, biological agents, or chemical weapons",
            "Illegal drugs, tobacco, alcohol, or cannabis",
            "Providing instructions or information on methods of self-harm",
            "Statements that encourage violence (considered violating under Violence)"
        ]
    }

    df = pd.DataFrame(data)
    st.markdown("Llama Guard Custom Taxonomy")
    st.table(df)