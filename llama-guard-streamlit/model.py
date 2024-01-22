from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from template import unsafe_categories

model_id = "meta-llama/LlamaGuard-7b"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

def moderate_with_template(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def moderation_prompt_for_chat(chat):
    # For simplicity, we assume the chat dictionary correctly alternates "user" and "assistant" roles
    # Feel free to include error checking if needed
    conversation = [turn["content"] for turn in chat]
    is_agent = len(conversation) % 2 == 0
    role = "Agent" if len(conversation) % 2 == 0 else "User"
    prompt = f"""[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.

    <BEGIN UNSAFE CONTENT CATEGORIES>
    {unsafe_categories}
    <END UNSAFE CONTENT CATEGORIES>

    <BEGIN CONVERSATION>

    """

    # Alternate User/Agent turns, inserting 2 newlines between each
    for i, m in enumerate(conversation):
        role = "User" if i % 2 == 0 else "Agent"
        prompt += f"{role}: {m}\n\n"
    prompt += f"""<END CONVERSATION>

Provide your safety assessment for {role} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
    return prompt

def moderate_chat(chat):
    prompt = moderation_prompt_for_chat(chat)
    inputs = tokenizer([prompt], return_tensors='pt').to("cuda")
    output = model.generate(**inputs, max_new_tokens=100, pad_token_id=0)
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)