import os
#os.environ["GEVENT_SUPPORT"] = "True"
#from google.colab import drive
#drive.mount('/content/drive/')
#os.chdir('/content/drive/MyDrive/Group36-INLPT-WS2023-Final/Group36-INLPT-WS2023-Final/')
from data.databases import NumpyDataBase
from data.embedding_models import EmbeddingModelMiniLML6
from training.utils import LLAMA_TEMPLATES, MISTRAL_TEMPLATES, system_message, format_user_message, format_conversation
import json


with open("data/data.json", "r") as f:
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
}["mistral"]

from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
torch.cuda.empty_cache()
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model['path'],
    quantization_config=quant_config,
    #trust_remote_code=True
)
model.config.use_cache = False


#model = PeftModel.from_pretrained(model, f"training/{base_model['save-path']}/")
#model = prepare_model_for_kbit_training(model)

tokenizer  = AutoTokenizer.from_pretrained(base_model['path'], use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

from ragatouille import RAGPretrainedModel
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.llms import LLM
from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import json
import datasets

def answer_with_rag(
    question: str,
    llm: LLM,
    knowledge_index: VectorStore,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 7,
) -> Tuple[str, List[str]]:
    relevant_docs = database.retrieve_by_query(question, top=num_retrieved_docs)
    relevant_docs = [doc for doc in relevant_docs]

    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, top=num_docs_final)
        relevant_docs = [doc for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]
    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": format_user_message(question, relevant_docs)}
    ]
    _, input_ids = format_conversation(conversation, base_model['templates'], tokenizer, training=False)
    print(tokenizer.decode(input_ids))
    len_prompt = len(input_ids)
    input_ids=torch.tensor(input_ids).unsqueeze(0)
    response = model.generate(input_ids, max_new_tokens=512).squeeze().tolist()
    answer = tokenizer.decode(response[len_prompt:-1])
    return answer, relevant_docs


from langchain_core.language_models import BaseChatModel
def run_rag_tests(
    eval_dataset: datasets.Dataset,
    llm: BaseChatModel,
    knowledge_index: VectorStore,
    output_file: str,
    reranker: Optional[RAGPretrainedModel] = None,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for i in tqdm(range(len(eval_dataset["question"]))):

        question = eval_dataset["question"][i]
        if question in [output["question"] for output in outputs]:
            continue

        answer, relevant_docs = answer_with_rag(
            question, llm, knowledge_index, reranker=reranker
        )
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {eval_dataset["answer"][i]}')
        result = {
            "question": question,
            "true_answer": eval_dataset["answer"][i],

            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)

EVALUATION_PROMPT2 = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""
EVALUATION_PROMPT = """
Task Description: An instruction (might include an Input inside it), a query with response, context, and a score rubric representing evaluation criteria are given.
1. You are provided with evaluation task with the help of a query with response and context.
2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general.
3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric.
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] YES or NO”
5. For example, Feedback: Some words. [RESULT] NO
6. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.
###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
Score YES: If the response for the query is in line with the context information provided.
Score NO: If the response for the query is not in line with the context information provided.
"""

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage


evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)

system_message = "You are an assistant that answers medical questions. You are presented with a user message aswell as with some document snippets that were deemed to be potentially relevant to the user's message. Using the information contained in the context,give a comprehensive answer to the question.Respond only to the question asked, response should be concise and relevant to the question. Provide the number of the source document when relevant. If the answer cannot be deduced from the context, do not give an answer, if you do give the answer, please source your answer as well"

from langchain.chat_models import ChatOpenAI
# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os
from torch import cuda, bfloat16
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
'''model_id = 'meta-llama/Llama-2-7b-chat-hf'
evaluator_name = "BAAI"'''
from accelerate.utils import BnbQuantizationConfig
import torch



eval_chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0,openai_api_key='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')
evaluator_name = "GPT3.5"
'''
eval_chat_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
evaluator_name = "GPT4"
'''

'''def evaluate_answers(
    answer_path: str,
    generate_text,
    evaluator_name: str,
    evaluation_prompt_template: ChatPromptTemplate,
) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        messages_str = [str(message) for message in eval_prompt]

        # Join the messages into a single string
        messages_combined = "\n".join(messages_str)
        print(messages_combined)
        eval_result = generate_text(messages_combined)
        eval_result = eval_result['generated_text']
        feedback, score = [
            item.strip() for item in eval_result.content.split("[RESULT]")
        ]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f)'''

def evaluate_answers(
    answer_path: str,
    eval_chat_model: BaseChatModel,
    evaluator_name: str,
    evaluation_prompt_template: ChatPromptTemplate,
) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        print(eval_result)
        feedback, score = [
            item.strip() for item in eval_result.content.split("[RESULT]")
        ]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f)

import os
'''os.environ["GEVENT_SUPPORT"] = "True"
from google.colab import drive
drive.mount('/content/drive/')
os.chdir('/content/drive/MyDrive/Group36-INLPT-WS2023-Final/Group36-INLPT-WS2023-Final/')
'''
import json
from data.databases import NumpyDataBase
from data.embedding_models import EmbeddingModelMiniLML6
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import os
from langchain.docstore.document import Document as LangchainDocument


if not os.path.exists("./output"):
    os.mkdir("./output")

with open("data/data.json", "r") as f:
    data = json.load(f)
embedding_model = EmbeddingModelMiniLML6()
knowledge_index = NumpyDataBase(data=data, embedding_model=embedding_model)
with open("QA_Generated_EmbeddingModelMiniLML6.json", "r") as f:
   eval_dataset = json.load(f)

for chunk_size in [200]:  # Add other chunk sizes (in tokens) as needed
    for embeddings in ["all-MiniLM-L6-v2"]:  # Add other embeddings as needed
        for rerank in [True, False]:
            settings_name = f"chunk{chunk_size}_embeddings{embeddings.replace('/', '~')}_rerank{rerank}_reader-model{base_model['path']}"
            output_file_name = f"output/rag_{settings_name}.json"

            print(f"Running evaluation for {settings_name}:")

            print("Loading knowledge base embeddings...")

            print("Running RAG...")
            '''reranker = (
                RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
                if rerank
                else None
            )'''
            run_rag_tests(
                eval_dataset=eval_dataset,
                llm=model,
                knowledge_index=knowledge_index,
                output_file=output_file_name,
                reranker=None,
                verbose=False,
                test_settings=settings_name,
            )

print("Running evaluation...")
evaluate_answers(
    output_file_name,
    eval_chat_model,
    evaluator_name,
    evaluation_prompt_template,
)

import glob

outputs = []
for file in glob.glob("output/*.json"):

    output = pd.DataFrame(json.load(open(file, "r")))
    output["settings"] = file
    outputs.append(output)

result = pd.concat(outputs)

result["eval_score_GPT3.5"] = result["eval_score_GPT3.5"].apply(
    lambda x: int(x) if isinstance(x, str) else 1
)
result["eval_score_GPT3.5"] = (result["eval_score_GPT3.5"] - 1) / 4

average_scores = result.groupby("settings")["eval_score_GPT3.5"].mean()
average_scores.sort_values()