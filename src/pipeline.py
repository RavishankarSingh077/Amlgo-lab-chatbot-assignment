from src.retriever import get_retriever
from src.generator import get_llm, get_qa_chain, get_prompt

def build_pipeline(vectorstore):
    retriever = get_retriever(vectorstore)
    llm = get_llm()
    prompt = get_prompt()
    qa_chain = get_qa_chain(llm, prompt)
    return retriever, qa_chain
