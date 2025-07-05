from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_ollama import OllamaLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_llm():
    return OllamaLLM(model="phi", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

def get_qa_chain(llm, prompt):
    return load_qa_with_sources_chain(llm=llm, chain_type="stuff", prompt=prompt, document_variable_name="context")

def get_prompt():
    template = """You are an intelligent assistant...
    Context:{context}
    Question:{question}
    Answer:"""
    return PromptTemplate(input_variables=["context", "question"], template=template)
