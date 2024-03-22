import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from dotenv import load_dotenv


# read the pdf files and extract text from files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        print(pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# convert text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_stores(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectore_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectore_store.save_local('faiss_index')

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in the provided context just say,"answer is not avilable in the context", don't provide the 
    wrong answer \n\n
    Context: \n {context}?\n
    Question: \n {question}?\n

    Answer: 

    """
    llm=CTransformers(model='models/llama-2-7b-ggml-model-f16-q4_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature': 0.01,
                              'gpu_layers':1})
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local('faiss_index', embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {'input_documents': docs, 'question': user_question},
        return_only_outputs=True
    )
    print(response)
    st.write('Reply: ', response['output_text'])

def main():
    st.set_page_config('Chat with Multiple PDF')
    st.header('Chat with multiple PDF using Gemini')

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title('Menu:')
        pdf_docs = st.file_uploader('Upload your PDF files and click on submit', accept_multiple_files=True)
        if st.button('Submit & Process'):
            with st.spinner('Processing...'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_stores(text_chunks)
                st.success('Done')

if __name__ == '__main__':
    main()