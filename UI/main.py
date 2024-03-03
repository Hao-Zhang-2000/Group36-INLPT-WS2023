import requests
from taipy.gui import Gui, State, notify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
import os
from data.databases import NumpyDataBase
from data.embedding_models import EmbeddingModelMiniLML6
from training.utils import LLAMA_TEMPLATES, MISTRAL_TEMPLATES, system_message, format_user_message, format_conversation
import json



context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI-Assistant. How can I help you today? "
conversation = {
    "Conversation": ["Who are you?", "Hi! I am AI-Assistant. How can I help you today?"]
}
current_user_message = ""

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
headers = {"Authorization": "Bearer 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"}



def request(state: State, prompt: str) -> str:
    """
    Send a prompt to the HuggingFace API and return the response.

    Args:
        - state: The current state of the app.
        - prompt: The prompt to send to the API.

    Returns:
        The response from the API.
    """

    user_query = "How can cold social intelligence promote behaviors in girls that lead to social preference?"
    relevant_abstracts = database.retrieve_by_query(user_query)

    conversation = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": format_user_message(user_query, relevant_abstracts)
        }
    ]
    _, input_ids = format_conversation(conversation, base_model['templates'], tokenizer, training=False)

    response = model.generate(input_ids=torch.tensor(input_ids).unsqueeze(0), max_new_tokens=1)

    return tokenizer.decode(response.squeeze().tolist())

def send_message(state: State) -> None:
    """
    Send the user's message to the API and update the conversation.

    Args:
        - state: The current state of the app.
    """
    # Add the user's message to the context
    state.context += f"Human: \n {state.current_user_message}\n\n AI:"
    # Send the user's message to the API and get the response
    answer = request(state, state.context).replace("\n", "")
    # Add the response to the context for future messages
    state.context += answer
    # Update the conversation
    conv = state.conversation._dict.copy()
    conv["Conversation"] += [state.current_user_message, answer]
    state.conversation = conv
    # Clear the input field
    state.current_user_message = ""

def style_conv(state: State, idx: int, row: int) -> str:
    """
    Apply a style to the conversation table depending on the message's author.

    Args:
        - state: The current state of the app.
        - idx: The index of the message in the table.
        - row: The row of the message in the table.

    Returns:
        The style to apply to the message.
    """
    if idx is None:
        return None
    elif idx % 2 == 0:
        return "user_message"
    else:
        return "gpt_message"


page = """
<|{conversation}|table|show_all|style=style_conv|>
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|>
"""

if __name__ == "__main__":
    with open("../data/data.json", "r") as f:
        data = json.load(f)
    embedding_model = EmbeddingModelMiniLML6()
    database = NumpyDataBase(data=data, embedding_model=embedding_model)

    base_model = {
        "llama": {
            "path": "meta-llama/Llama-2-7b-chat-hf",
            "save-path": "MediRAG-LLaMA",
            "templates": LLAMA_TEMPLATES
        },
        "mistral": {
            "path": "mistralai/Mistral-7B-Instruct-v0.2",
            "save-path": "MediRAG-Mistral",
            "templates": MISTRAL_TEMPLATES
        },
        "meditron": {
            "path": "epfl-llm/meditron-7b",
            "save-path": "MediRAG-Meditron2",
            "templates": LLAMA_TEMPLATES
        }
    }["llama"]

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model['path'],
        quantization_config=quant_config,
        cache_dir='I:\\huggingfacemodels',
        use_auth_token='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'
        # trust_remote_code=True
    )
    model.config.use_cache = False

    model = PeftModel.from_pretrained(model, f"../training/{base_model['save-path']}/")
    # model = PeftModel.from_pretrained(model)
    # model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model['path'], use_fast=False,
                                              use_auth_token='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')
    tokenizer.pad_token = tokenizer.eos_token

    Gui(page).run(dark_mode=True, title="Taipy Chat")
