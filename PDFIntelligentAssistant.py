
import streamlit as st
import os

from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS # vector database localy

import google.generativeai as genai

from dotenv import load_dotenv #to access the API key in .env


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key= os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000,
        chunk_overlap = 2000,
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
   
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,)

    prompt = PromptTemplate(template =prompt_template, input_variables = ["context","question"] )

    chain = load_qa_chain(model, chain_type="stuff",prompt = prompt)

    return chain

def user_input(user_q):
    
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_q)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents" : docs, "question": user_q},
        return_only_outputs = True
    )

    
    return response

def main():

    st.set_page_config(page_title = "Chat with multiple PDFs", page_icon = ":books")
    
    st.header("Chat with multiple PDFs")

    user_question = st.text_input("Ask a question about your documents:")

    
    if user_question:
        response = user_input(user_question)
        st.write("Reply: ", response["output_text"])


    with st.sidebar:
        st.subheader("Your documanets")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Proccessing"):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)
        
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store, embeddings, semantic search
                get_vectorstore(text_chunks)


                st.success("Done")




  

if __name__ == '__main__':
    main()