from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

def build_qa_chain(vector_store):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=512
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template(
        """
You are a helpful assistant.

Use ONLY the provided context to answer the question.
If the answer is not present in the context, say:
"I don't have enough information in the provided documents to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # LCEL pipeline (no deprecated imports)
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
