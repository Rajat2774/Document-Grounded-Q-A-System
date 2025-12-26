from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gc

def load_pdf(path):
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len
        )

        chunks=splitter.split_documents(docs)

        del loader
        del docs
        gc.collect()
        return chunks
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise
    finally:
        # Ensure garbage collection
        gc.collect()
