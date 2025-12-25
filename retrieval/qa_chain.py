from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()


def build_qa_chain(vector_store):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 6}
    )

    prompt_template = """
Use the following pieces of context to answer the question.
If the answer is present in the context, provide a detailed explanation.
If the answer cannot be found, say:
"I don't have enough information in the provided documents to answer this question."

Context:
{context}

Question:
{input}

Detailed Answer:
"""

    prompt = PromptTemplate.from_template(prompt_template)

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    return retrieval_chain
