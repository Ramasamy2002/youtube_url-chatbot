from PyPDF2 import PdfFileWriter, PdfReader
import streamlit as st
from llama_index.core.node_parser import SentenceSplitter


def extract_text(files):
    if files is not None:
        content=""
        for pdf in files:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
    return content
def extract_chunks(content):
    splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,)
    nodes = splitter.get_nodes_from_documents(content)
    return nodes
    

# with st.sidebar:
st.title("Menu:")
pdf_docs = st.file_uploader(
    "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
if st.button("Submit & Process"):
    extracted_text=extract_text(pdf_docs)
    extracted_chunks=extract_chunks([extracted_text])
    st.write(extracted_chunks)
