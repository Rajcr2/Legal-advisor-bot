# Legal-Advisor

## Introduction
I have built an **Agentic Legal Question-Answering System** using an agentic RAG approach.
The system performs multi-step ReAct reasoning, retrieves and then filters legal context, generates grounded answers with respect to query and also, evaluates QA system performance using structured semantic and retrieval-based metrics.
The system is exposed both as a REST API and as a MCP tool, so it can be used by users through UI as well as by other AI systems.

## Overview

The system behaves like a **Semi-Autonomous ReAct-style AI agent** that is capable of :

- **Understanding legal questions**
- **Planning multi-step reasoning**
- **Retrieving relevant legal documents with respect to questions**
- **Filtering useful context from retrieved context**
- **Generating grounded answers**

Along with this, system also includes :

- **A REST API (FastAPI) layer for the agent's external access**  
- **An FastMCP server for other AI systems so, they can call it as a tool**
- **A ReAct based chat UI for interactive querying**
- **DeepEval based evaluation pipeline to evaluate answer quality**

## Key Features

- **LangGraph agent workflow**
- **ReAct-style reasoning** (Think → Act → Observe → Answer)
- **Retrieval-Augmented Generation** (RAG)
- **Vector-based semantic search**
- **LLM based retrieved Context filtering** (Selector)
- **Grounded answers**
- **Multi-step reasoning loop**
- **FastAPI-based REST API**
- **MCP server for AI-to-AI integration**
- **Evaluation framework using DeepEval**
- **Golden dataset regression testing**

## Agent Workflow

The agent follows a **ReAct-style autonomous loop** :

1. **Think** - analyze query & decide next action
2. **Act** - retrieve legal context from vector database
3. **Observe** - evaluate retrieved information
4. **Repeat** - until sufficient knowledge obtained
5. **Answer** - generate final grounded response

The agent stops naturally when retrieved knowledge stabilizes.

## Tech Stack

| Layer           | Tech                            |
| --------------- | ------------------------------- |
| **Agent**       | LangGraph (ReAct pattern)       |
| **LLM**         | Mistral 7B via Ollama           |
| **RAG**         | ChromaDB + SentenceTransformers (all-MiniLM-L6-v2) |
| **Backend**     | FastAPI, Pydantic                         |
| **MCP**         | FastMCP                         |
| **Frontend**    | React (Vite)                    |
| **Storage**     | PostgreSQL (PDFs)               |
| **Evaluation**  | DeepEval                        |

### Prerequisites
To run this project, you need to install the following libraries :

### Required Libraries

- **Python 3.12+**
- **PostgreSQL**: PostgreSQL is a powerful, open-source object-relational database system known for its reliability along with SQL compliance.
- **ChromaDB**: ChromaDB is an open-source embedding database designed for storing, querying, and retrieving vector embeddings for RAG applications.
- **Ollama**: Ollama is a lightweight tool that lets you run large language models (LLMs) like Mistral-7B, llama3 locally.
- **LangGraph**: LangGraph is a framework for building stateful, multi-step AI agent workflows. It enables controlled reasoning, tool orchestration, and conditional execution using graph-based pipelines.
- **Node.js**: Node is a JavaScript runtime for building scalable backend apps.
- **Deepeval**: DeepEval is an evaluation framework for LLM and RAG systems that measures answer quality, faithfulness, hallucination, retrieval relevance, and overall task performance using structured metrics.

Other Utility Libraries : **psycopg2**, **textwrap**.

### Installation

   ```
   ollama serve
   ollama run mistral
   pip install langgraph
   pip install psycopg2-binary
   pip install chromadb
   pip install deepeval

   cd frontend
   npm install
   ```

### API

| Method | Endpoint    | Use                     |
| ------ | ----------- | ----------------------- |
| GET    | /health     | Check if API is running |
| POST   | /api/v1/ask | Ask legal question      |

**Docs**: http://localhost:8000/docs

### MCP Tool

The agent is also exposed as a tool:
   ```
   ask_legal_question(question: str) -> str
   ```
This allows tools like Claude Desktop or Cursor to call it directly.


### Procedure

1.   Create new directory **'Legal Advisor'**.
2.   Inside that directory/folder create new environment.
   
   ```
   python -m venv legaladv
   ```

  Now, activate this **'legaladv'** venv.
  
4.   Clone this Repository :

   ```
   git clone https://github.com/Rajcr2/LegalAdvisor.git
   ```
5.   Now, Install all mentioned required libraries in your environment.
6.   Firstly Store legal documents in PostgreSQL for that run following command.
   ```
   python Store.py
   ``` 
   When pdfs are succefully stored you will get output like below :

   <img width="1920" height="1080" alt="Store py" src="https://github.com/user-attachments/assets/d313727c-7d69-4600-aacb-25307e80ee16" />

   <img width="1920" height="1080" alt="Store Postgre" src="https://github.com/user-attachments/assets/30c64697-6ff7-41ef-8b1f-02e8b4ba877b" />

________

7.   Lets, generate vector Embeddings now. for that, Run following command :
   
    python Embeddings.py

<img width="1920" height="1080" alt="Embeddings py" src="https://github.com/user-attachments/assets/f5f8e812-4efa-4699-8830-0782e960532e" />

________

8.   Now, we are almost done. Now, start backend server.
     Run Uvicorn to start the FastAPI server from terminal. This will start the backend API.     

     ```
     uvicorn api:app --reload
     ``` 
      

8.   Next step is in another, Terminal we have start the frontend UI. Firstly we have to navigate to frontend folder and run the developement server using npm.
     This will open chatbot in browser at **http://localhost:5173** .

     ```
     cd frontend
     npm run dev
     ``` 

Ask any legal question like this **"My employer ignored my complaint, what are my rights ? Under POSH Act ?"** and get the **'legal Advice'**.


### Output

https://github.com/user-attachments/assets/68067416-694a-4201-81de-2443ca277dbd
_______

This was about **Agentic System**. Tested the same system using DeepEval's pre-built metrics below is the more about it.

## Evaluation Framework

**Metrics Used** :
- **Answer Relevancy**
- **Faithfulness**
- **Hallucination**
- **Context Relevancy**
- **Context Precision**
- **Task Completion**
- **Tool Correctness**

Evaluation uses a **Golden Dataset** for repeatable testing and regression detection.

<img width="1446" height="606" alt="golden dataset eval" src="https://github.com/user-attachments/assets/e16d87f8-c97f-4c57-a323-b4f6debb189f" />


Run **'python Evaluation.py'** file from Terminal.
   
    python Evaluation.py

1. **First it create Test cases like this.**

<img width="1920" height="1080" alt="Test case 1" src="https://github.com/user-attachments/assets/11ac9e7c-fa13-4e2c-bf2a-4e9f2e5ac816" />


2. **Then evaluation starts.**

<img width="1383" height="316" alt="running main" src="https://github.com/user-attachments/assets/43a6979b-a772-49ed-a9ac-7fa1f87d4266" />

3. **as evaluation progresses, we get Test cases Results like this for All Test cases and in the end Final Evaluation Output.**


#### Test Case 1
<img width="1920" height="1080" alt="evaluation tc 1" src="https://github.com/user-attachments/assets/31204a76-20d0-4a9b-8608-b802be20ead9" />

#### Test Case 2
<img width="1920" height="1080" alt="evaluation tc 2" src="https://github.com/user-attachments/assets/c674e1bb-6f47-42af-a45e-9f8a5aecf4e2" />

#### Test Case 3
<img width="1920" height="1080" alt="evaluation tc 3" src="https://github.com/user-attachments/assets/59db4f34-cbd5-4cb1-84ee-54b0796d4851" />

#### Test Case 4
<img width="1920" height="1080" alt="evaluation tc 4" src="https://github.com/user-attachments/assets/560bbf46-dbf2-47f0-8730-f24141bae7f8" />


## Evaluation Output

<img width="1920" height="1080" alt="Evaluation final" src="https://github.com/user-attachments/assets/3f424cf1-59f4-4400-b535-3acfcd72850f" />


_____


### What Makes This System Agentic?

- **LLM decides next action dynamically**
- **Multi-step reasoning loop with state tracking**
- **Tool usage (Retriever, Selector, Answer)**
- **Autonomous stopping until it has enough information**
- **Non-deterministic decision making**
- **Fully testable agent execution pipeline**

##  Final Thoughts

This **Legal Advisor** was mainly about to move from a simple LLM app to **"Mini AI Legal Advocate"** type something more agent-like and production-ready system 
that combines **retrieval, contextual reasoning, APIs and evaluation** into one system — and can be used both by users and other AI tools.

