from utils import LLAMA_TEMPLATES, MISTRAL_TEMPLATES
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
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = {
    "llama": {
        "path": "meta-llama/Llama-2-7b-chat-hf"
        "templates": LLAMA_TEMPLATES
    },
    "mistral": {
        "path": "mistralai/Mistral-7B-Instruct-v0.2",
        "templates": MISTRAL_TEMPLATES
    },
    "meditron": {
        "path": "epfl-llm/meditron-7b",
        "templates": LLAMA_TEMPLATES
    }
}["meditron"]

model, message_templates = load_model(
    base_model['path'], 
    quant_config=quant_config, 
    lora_config=lora_config,
    use_auth_token=credentials['huggingface']
)

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

training_params = TrainingArguments(
    output_dir="./FunctionTraining",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
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
    train_dataset=trainset,
    data_collator=lambda x: x
)

trainer.train()

trainer.save_model("MediRAG/")


