# LLM Security Project with Llama Guard

This repository provides a quick and easy way to run the Llama Guard application on your local machine and explore LLM security.

## What is Llama Guard?

Llama Guard is a defensive framework designed to detect and mitigate potential security risks associated with Large Language Models (LLMs). It helps developers and researchers build safer and more reliable LLM applications.


## What's included?

Nemo Guardrail Implementation: The llama-guard folder contains a NeMo Guardrail implementation, offering flexibility and customization for your specific needs.
Streamlit Applications: Two Streamlit applications are provided for convenient testing:
llama-guard-only.py: Test input prompts and responses directly with Llama Guard.
llama_2_with_llama-guard.py: Run Llama Guard with the pre-trained Llama 2 13b model for real-world testing.

## How to run the application:

- Clone this repository
- Install dependencies: pip install -r requirements.txt
- Run the desired application:
  1. Test Llama Guard: streamlit run llama-guard-only.py
  2. Test with Llama 2 13b: streamlit run llama_2_with_llama-guard.py


## Learn More:

Blog post: Deepen your understanding of Llama Guard and LLM security with this informative blog post: https://balavenkatesh.medium.com/securing-tomorrows-ai-world-today-llama-guard-defensive-strategies-for-llm-application-c29a87ba607f
