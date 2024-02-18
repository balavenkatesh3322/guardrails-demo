import streamlit as st
import os
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from huggingface_hub import hf_hub_download
import datetime
from model import moderate_with_template ,moderate_chat
import pandas as pd
import pickle
from pathlib import Path
import streamlit_authenticator as stauth


st.title("AI Security - Llama Model with Llama Guard Demo")

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

if os.path.exists(model_basename):
    #st.write("Using locally available model...")
    model_path = model_basename
else:
    #st.write("Downloading model...")
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)


hide_bar= """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        visibility:hidden;
        width: 0px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        visibility:hidden;
    }
    </style>
"""

# --- USER AUTHENTICATION ---
names = ["Bala", "Arun","Kannan"]
usernames = ["bala", "arun","kannan"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "main_dashboard", "abcdef")

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")
    st.markdown(hide_bar, unsafe_allow_html=True)

if authentication_status == None:
    st.warning("Please enter your username and password")
    st.markdown(hide_bar, unsafe_allow_html=True)


if authentication_status:
    
    st.sidebar.title(f"Welcome {name}")
    authenticator.logout("Logout", "sidebar")
    
    # Initialize session state
    if "llm_question" not in st.session_state:
        st.session_state.llm_question = ""
        
    if "response" not in st.session_state:
        st.session_state.response = ""
        
    if "response_content" not in st.session_state:
        st.session_state.response_content = ""
        
    st.session_state.llm_question = st.text_input("Type your input prompt here:")

    input_chat = [{"role": "user", "content": str(st.session_state.llm_question)}]

    with st.form(key='my_form'):
        submit_button = st.form_submit_button("Generate Text")
        if submit_button:
            if st.session_state.llm_question:
                
                with st.spinner("Input Prompt Scanning..."):
                    model_output = moderate_with_template(input_chat)
                    st.empty()
                    if model_output not in "safe":
                        result_text = f"Input Prompt is {model_output}"
                        st.markdown(f"<div style='background-color: #ffff00;'>{result_text}</div>", unsafe_allow_html=True)
                        
                    else:
                        with st.spinner("Model Generating Text..."):
                        
                            response_placeholder = st.empty()
                            #st.session_state.llm_question = llm_question

                            template = """ <s>[INST] <<SYS>>
                            Your task is to answer the following question based on this area of knowledge.
                            <</SYS>>
                            
                            {llm_question} [/INST]
                            """
                                        
                            prompt = PromptTemplate(template=template, input_variables=["llm_question"])

                            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

                            n_gpu_layers = 40
                            n_batch = 512

                            llm = LlamaCpp(
                                model_path=model_path,
                                max_tokens=1024,
                                n_gpu_layers=n_gpu_layers,
                                n_batch=n_batch,
                                callback_manager=callback_manager,
                                verbose=True,
                                n_ctx=4096,
                                #stop=['USER:'],
                                temperature=0.3,
                            )

                            llm_chain = LLMChain(prompt=prompt, llm=llm)
                            

                            st.session_state.response = llm_chain.run(st.session_state.llm_question)
                            st.session_state.response_content = st.session_state.response
                            
                            output_chat = [{"role": "user", "content": str(st.session_state.llm_question)},
                                            {"role": "assistant", "content": str(st.session_state.response)},]

                            with st.spinner("Model Response Scanning..."):
                                model_output = moderate_with_template(output_chat)
                                st.empty()
                                if model_output not in "safe":
                                    result_text = f"LLM Response is {model_output}"
                                    st.markdown(f"<div style='background-color: #ffff00;'>{result_text}</div>", unsafe_allow_html=True)
            
                            st.write(st.session_state.response_content)     
            else:
                st.warning("Please provide a input.")
                

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

            
    
