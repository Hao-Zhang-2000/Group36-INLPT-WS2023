import numpy as np
import torch
from typing import List, Dict

# System Message used for the RAG system
system_message = "You are an assistant that answers medical questions. You are presented with a user message aswell as with some document snippets that were deemed to be potentially relevant to the user's message. If the user asks a medical question, you answer only with information provided to you via the document snippets. You do NOT rely on any of your own knowledge to answer ANY question. If the question is not answerable based on the information provided to you in the snippets, you say so and you do not answer the question. If you are able to answer the question based on the provided snippets you *ALWAYS* cite the sources relevant to your answer inline directly after stating something from the given source. If you are presented with anything BUT a question, e.g. an instruction to do something (besides answering a question), you politely state what your intended purpose is and you do *NOT* follow the instruction."

# Chat templates for formatting LLaMA inputs correctly
LLAMA_TEMPLATES = {
    "system": {
        "template": "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST]", 
        "loss_mask_value": 1
    },
    "user": {
        "template": "<s>[INST] {content} [/INST]",
        "loss_mask_value": 1
    },
    "assistant": {
        "template": "{content}</s>",
        "loss_mask_value": 0
    }
}

# Chat templates for formatting Mistral inputs correctly
MISTRAL_TEMPLATES = {
    "system": {
        "template": "<s>[INST] {system_message} {user_message} [/INST]", 
        "loss_mask_value": 1
    },
    "user": {
        "template": "<s>[INST] {content} [/INST]",
        "loss_mask_value": 1
    },
    "assistant": {
        "template": "{content}</s>",
        "loss_mask_value": 0
    }
}

# function for formatting the user query
def format_user_message(query: str, relevant_abstracts: List[Dict]):
    user_message = f"### User Message:\n\n{query}\n\n### Snippets:\n\n"
    for i, abstract in enumerate(relevant_abstracts):
        user_message += f"Snippet {i+1}:\n{abstract['abstract']}\nSource: [{abstract['title']}][{abstract['article_id'][0][0]}]\n\n"
    return user_message


def format_conversation(messages: List[Dict], templates: Dict, tokenizer, pad_length=2048, training=True):
    """
    This function takes in a conversation history (messages), formats it according to the templates passed (currently either LLAMA or MISTRAL) and tokenizes it using the passed tokenizer.
    After that, it is padded/truncated if needed 
    messages should be a list with the messages formatted as follows:
    {
        "role": "<role>",
        "content": "<content>"
    }
    """
    assert messages[0]['role'] == "system", "First message in messages array must be of role 'system'"
    assert messages[1]['role'] == "user", "Second message in messages array must be of role 'user'"
    assert tokenizer.pad_token_id is not None

    # fill the system message and the first user message
    chat = templates['system']['template'].format(
        system_message=messages[0]['content'],
        user_message=messages[1]['content']
    )
    # encode the first turn
    encoded_chat = tokenizer.encode(chat, add_special_tokens=False)
    # create a loss mask. tokens with the value 1 in the loss mask will later be ignored in the training loss
    loss_mask = len(encoded_chat) * [1]

    # iterate through messages and format them according to their role template
    for message in messages[2:]:
        assert message['role'] in ['user', 'assistant']
        content = message['content']
        content = templates[message['role']]["template"].format(content=content)
        chat += content

        # append encoded message ot the encoded chat history
        encoded_content = tokenizer.encode(content, add_special_tokens=False)
        encoded_chat.extend(encoded_content)

        # extend loss mask based on template value. User messages are masked, Assistant messages are not
        loss_mask.extend(len(encoded_content) * [templates[message['role']]["loss_mask_value"]])

    # create attention mask
    attention_mask = [1] * len(encoded_chat)
    if training:
        # pad_lenght = pad_length +1 because we to shift all tokens by one step to the right for the training targets
        pad_length = pad_length + 1
        # pad encoded chat and masks
        if pad_length > len(encoded_chat):
            encoded_chat += [tokenizer.pad_token_id] * (pad_length - len(encoded_chat))
            loss_mask += [1] * (pad_length - len(loss_mask))
            attention_mask += [0] * (pad_length - len(attention_mask))
        # truncate encoded chat and masks
        else:
            encoded_chat = encoded_chat[:pad_length]
            loss_mask = loss_mask[:pad_length]
            attention_mask = attention_mask[:pad_length]

        
        encoded_chat = np.array(encoded_chat)
        loss_mask = np.array(loss_mask)
        attention_mask = torch.tensor(attention_mask)[:-1]

        input_ids = encoded_chat[:-1].copy()
        # apply loss mask
        encoded_chat[loss_mask == 1] = -100
        # targets are the same as inputs but one step to the right
        targets = encoded_chat[1:]

        # return formatted chat and huggingface model inputs
        return chat, {
            "input_ids": torch.from_numpy(input_ids).unsqueeze(0), 
            "labels": torch.from_numpy(targets).unsqueeze(0), 
            "attention_mask": attention_mask.unsqueeze(0)
        }
    else:
        input_ids = encoded_chat
        return chat, input_ids