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


st.title("AI Security - Llama Guard Demo")

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
names = ["bala", "arun","kannan"]
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
    
    # Initialize session state
    if "llm_question" not in st.session_state:
        st.session_state.llm_question = ""

    if "llama_guard_template_input_clicked" not in st.session_state:
        st.session_state.llama_guard_template_input_clicked = False

    if "llama_guard_template_output_clicked" not in st.session_state:
        st.session_state.llama_guard_template_output_clicked = False
        
    if "response" not in st.session_state:
        st.session_state.response = ""
        
    if "response_content" not in st.session_state:
        st.session_state.response_content = ""
        
    st.session_state.llm_question = st.text_input("Type your input prompt here:")

    with st.form(key='my_form'):
        submit_button = st.form_submit_button("Generate Text")
        if submit_button:
            if st.session_state.llm_question:
                #st.write("Generating response...")
                with st.spinner("Processing..."):
                
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
                
            
        # else:
        #     st.warning("Please provide a input.")
    if st.session_state.response_content:
        st.markdown("<h2 style='text-align: center; color: #666;'>Model Response</h2>", unsafe_allow_html=True)
        st.write(st.session_state.response_content)

    input_chat = [{"role": "user", "content": str(st.session_state.llm_question)}]

    output_chat = [
        {"role": "user", "content": str(st.session_state.llm_question)},
        {"role": "assistant", "content": str(st.session_state.response)},
    ]

    with st.expander("Llama Guard Template", expanded=False):
        if st.button("Scan Input Prompt",key="llama_guard_template_input"):
            if st.session_state.llm_question:
                #st.write("Generating response...")
                with st.spinner("Processing..."):
                    #st.write()
                    model_output = moderate_with_template(input_chat)
                    st.markdown(f"<div style='background-color: #ffff00;'>{model_output}</div>", unsafe_allow_html=True)
                    st.session_state.llama_guard_template_input_clicked = False
            else:
                st.warning("Please provide a input Prompt.") 


        if st.button("Scan Model Response",key="llama_guard_template_output"):
            if st.session_state.llm_question:
                #st.write("Generating response...")
                with st.spinner("Processing..."):
                    model_output = moderate_with_template(output_chat)
                    st.markdown(f"<div style='background-color: #ffff00;'>{model_output}</div>", unsafe_allow_html=True)
                    st.session_state.llama_guard_template_output_clicked = False
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
            if st.session_state.llm_question:
                #st.write("Generating response...")
                with st.spinner("Processing..."):
                    #st.write()
                    model_output = moderate_chat(input_chat)
                    st.markdown(f"<div style='background-color: #ffff00;'>{model_output}</div>", unsafe_allow_html=True)
            else:
                st.warning("Please provide a input Prompt.") 


        if st.button("Scan Model Response",key="llama_guard_fine_tune_template_output"):
            if st.session_state.llm_question and st.session_state.response:
                #st.write("Generating response...")
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
            
    
