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

st.title("AI Security - Llama Guard Demo")

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

if os.path.exists(model_basename):
    #st.write("Using locally available model...")
    model_path = model_basename
else:
    #st.write("Downloading model...")
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

    
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
                if model_output not in "safe":
                    st.markdown(f"<div style='background-color: #ffff00;'>{model_output}</div>", unsafe_allow_html=True)
                    
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
                            if model_output not in "safe":
                                st.markdown(f"<div style='background-color: #ffff00;'>{model_output}</div>", unsafe_allow_html=True)
        
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

            
    