## Group Members & Contributions
Name: Jatin Jatin<br />
Matriculation number: 4732232<br />
Course of study: Master’s in Data and Computer Science <br />
Mail ID: jatinrk21@gmail.com  <br />
Github ID:bhardwajjjatin <br />
wrote the report <br />

Name: Christopher Lindenberg<br />
Matriculation Number: 3715220<br />
Course of study: Bachelors in Computer Science 100%<br />
Mail: christopher.lindenberg@stud.uni-heidelberg.de <br />
Github ID:Christopher Lindenberg<br />
made the retrieval pipeline with embedding models and databases and tried to train some models but that kinda went wrong<br />

Name: Yulin Liu<br />
Matriculation Number: 4735159<br />
Course of study: Master in Scientific Computing 100%<br />
Mail: tong54387@gmail.com<br />
Github ID:Lin9773<br />
did the Data Collection, UI and RAG implementation and evaluation<br />

Name: Hao-Zhang<br />
Matriculation Number: 4735257<br />
Course of study: Master in Scientific Computing 100%<br />
Mail: hao.zhang@stud.uni-heidelberg.de<br />
Github ID:Hao-Zhang-2000<br />
did the data embedding and storage in Pinecone, and evaluation part for the information retrieval and the RAG<br />

## Dependencies
see `requirements.txt`

## Before The Start
the `data.json` file in directory `data` is missing, to generate it, run the `embed_and_store.py` at the directory `data`

## UI
the `main.py` file located in `UI` folder starts the UI interface once finishing loading the model.

![Preview](preview1.png)

This directory contains the User Interface components of our project.

## Evaluation
the json for evaluation is already generated in folder `output`, to re-generate it, run the `evaluate.py` at the root folder. To evaluate the rag, run the `Evaluation/evaluate_method1.py`. To evaluate the IR, run `data/IR_evaluation_numpy.ipynb`





