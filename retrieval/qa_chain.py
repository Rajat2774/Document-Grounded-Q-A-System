from langchain_groq import ChatGroq
from langchain_community.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

def build_qa_chain(vector_store):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 6} 
    )
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you can find the answer in the context, provide a detailed answer.
    If you cannot find the answer in the context, say "I don't have enough information in the provided documents to answer this question."
    
    Context:
    {context}
    
    Question: {question}
    
    Detailed Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain
