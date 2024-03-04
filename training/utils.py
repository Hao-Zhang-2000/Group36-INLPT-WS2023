import numpy as np
import torch

system_message = "You are an assistant that answers medical questions. You are presented with a user message aswell as with some document snippets that were deemed to be potentially relevant to the user's message. Using the information contained in the context,give a comprehensive answer to the question.Respond only to the question asked, response should be concise and relevant to the question. Provide the number of the source document when relevant. If the answer cannot be deduced from the context, do not give an answer, if you do give the answer, please source your answer as well"

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

def format_user_message(query, relevant_abstracts):
    user_message = f"### User Message:\n\n{query}\n\n### Snippets:\n\n"
    for i, abstract in enumerate(relevant_abstracts):
        user_message += f"Snippet {i+1}:\n{abstract['abstract']}\nSource: [{abstract['title']}][{abstract['article_id'][0][0]}]\n\n"
    return user_message


def format_conversation(messages, templates, tokenizer, pad_length=2048, training=True):
    assert messages[0]['role'] == "system", "First message in messages array must be of role 'system'"
    assert messages[1]['role'] == "user", "Second message in messages array must be of role 'user'"
    assert tokenizer.pad_token_id is not None
    
    chat = templates['system']['template'].format(
        system_message=messages[0]['content'],
        user_message=messages[1]['content']
    )
    encoded_chat = tokenizer.encode(chat, add_special_tokens=False)
    loss_mask = len(encoded_chat) * [1]

    for message in messages[2:]:
        assert message['role'] in ['user', 'assistant']
        content = message['content']
        content = templates[message['role']]["template"].format(content=content)
        chat += content
        
        encoded_content = tokenizer.encode(content, add_special_tokens=False)
        encoded_chat.extend(encoded_content)
        loss_mask.extend(len(encoded_content) * [templates[message['role']]["loss_mask_value"]])
                         
    attention_mask = [1] * len(encoded_chat)
    if training:
        pad_length = pad_length + 1
        if pad_length > len(encoded_chat):
            encoded_chat += [tokenizer.pad_token_id] * (pad_length - len(encoded_chat))
            loss_mask += [1] * (pad_length - len(loss_mask))
            attention_mask += [0] * (pad_length - len(attention_mask))
        else:
            encoded_chat = encoded_chat[:pad_length]
            loss_mask = loss_mask[:pad_length]
            attention_mask = attention_mask[:pad_length]

        encoded_chat = np.array(encoded_chat)
        loss_mask = np.array(loss_mask)
        attention_mask = torch.tensor(attention_mask)[:-1]

        input_ids = encoded_chat[:-1].copy()
        encoded_chat[loss_mask == 1] = -100
        targets = encoded_chat[1:]
        return chat, {
            "input_ids": torch.from_numpy(input_ids).unsqueeze(0), 
            "labels": torch.from_numpy(targets).unsqueeze(0), 
            "attention_mask": attention_mask.unsqueeze(0)
        }
    else:
        input_ids = encoded_chat
        return chat, input_ids
