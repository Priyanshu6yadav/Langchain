import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

file_path = "/Users/priyanshuyadav/Desktop/Langchain/Text Splitter/Minor_Project_Synopsis_AI-SOC.pdf"

# Verify path before loading
if not os.path.exists(file_path):
    print(f"Error: Could not find '{file_path}' in {os.getcwd()}")
else:
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separator=''
    )

    # Note the plural: split_documents
    result = splitter.split_documents(docs)
    print(result[0].page_content )