import torch
from torch import nn
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from utils import LLAMA_TEMPLATES, MISTRAL_TEMPLATES, system_message
from dataset import RAGDataset
from tqdm import tqdm
import json
import yaml

with open('credentials.yaml', 'r') as file:
     credentials = yaml.safe_load(file)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inp = inputs[0]
        labels = inp.pop("labels")
        outputs = model(**inp)
        logits = outputs.get("logits")
        B, T, C = logits.shape
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(B*T, C), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
)

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
        "save-path": "MediRAG-Meditron",
        "templates": LLAMA_TEMPLATES
    }
}["mistral"]

save_path = base_model['save-path']
message_templates = base_model['templates']
model = AutoModelForCausalLM.from_pretrained(
    base_model['path'], 
    quantization_config=quant_config, 
    use_auth_token=credentials['huggingface']
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

tokenizer  = AutoTokenizer.from_pretrained(
    base_model['path'], 
    use_auth_token=credentials['huggingface'],
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token

dataset = RAGDataset(
    tokenizer,
    system_message,
    LLAMA_TEMPLATES
)
print("Samples in dataset:", len(dataset))

training_params = TrainingArguments(
    output_dir=f"./{save_path}-Training",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    save_steps=50,
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.,
    warmup_ratio=0.03,
    lr_scheduler_type="linear"
)

trainer = CustomTrainer(
    model=model,
    args=training_params,
    train_dataset=dataset,
    data_collator=lambda x: x
)

trainer.train()

trainer.save_model(f"{save_path}")
#trainer.push_to_hub("MediRAG-Meditron")

