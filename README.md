#  Amlgo Lab Chatbot Assignment #

## 📽️ Demo Video 

[Demo video link (Google drive]=(https://drive.google.com/drive/folders/1W1iTL9It494BlG9YLutFHKfAGIVq17E4?usp=drive_link )

[Demo video link (GitHub Repo]=(https://drive.google.com/drive/folders/1W1iTL9It494BlG9YLutFHKfAGIVq17E4?usp=drive_link )

## 📽️ GitHub Repo Link

[GitHub Repo project link]=(https://drive.google.com/drive/folders/1W1iTL9It494BlG9YLutFHKfAGIVq17E4?usp=drive_link )




This video demonstrates how the RAG-based document chatbot works with real-time streaming responses.


This is my project submission for the Junior AI Engineer Assignment by Amlgo Labs.

:In this project, I have built a smart chatbot that understands any given document and answers questions based on it. It uses Retrieval-Augmented Generation (RAG) and shows real-time answers using a Streamlit interface.

What This Chatbot Does:-
-Imagine you have a long legal or policy document. You can ask questions like:

-What is the return policy?

-When do my data rights apply?

-This chatbot reads the document and gives you accurate answers pulled directly from the document itself.

--How It Works (Architecture):-
User Query ➝ Retriever ➝ Vector Database ➝ Generator ➝ Streaming Answer

-First, the document is cleaned and split into smaller chunks
-Each chunk is converted into an embedding and stored in a vector database (Chroma)

-When a user asks a question, the retriever finds the most relevant chunks using semantic search

-These chunks, along with the user’s question, are sent to the language model

-The model generates the answer and streams it gradually like a real conversation

--How to Run This Project on Your System:-
Download or clone the code from GitHub

-Create a new virtual environment and activate it

-Install all required libraries using requirements.txt

-Create a file named .env and add your OpenAI API key in it

-Run the app using the command: streamlit run app.py

-The app will open in your browser and you can start asking questions

--Features:-
Users can type questions

-Answers are streamed step-by-step like a chat

-The source text is also shown with each answer to indicate where it came from

-The sidebar displays which model is used and how many chunks are processed

-Clear/reset button to restart the chatbot anytime

--Technologies Used:-
-Embedding Model: all-MiniLM-L6-v2

-Vector Database: Chroma

-Language Model: TinyLlama (via Ollama)

-Frontend UI: Streamlit

-Backend Logic and Prompting: LangChain

--Sample Queries:-

1. What is the refund condition?

2. Who can access my data?


--Project Folder Structure:-
-data: Original PDF document

-chunks: JSON files with processed text chunks

-vectordb: Files for Chroma vector database

-notebooks: Jupyter notebooks for preprocessing and evaluation

-src: Code for retriever, generator, and RAG pipeline

-app.py: The main Streamlit app

-requirements.txt: Required Python packages

-README.md: This project summary

-report.pdf: 2–3 page project report

Thank You
Thanks to Amlgo Labs for this assignment opportunity.
I learned a lot while building this project.
Looking forward to your valuable feedback.

This project was created by Ravishankar Singh.
