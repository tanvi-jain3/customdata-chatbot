import os
import glob
import shutil
import requests
from typing import List
from datetime import datetime
from bs4 import BeautifulSoup
import streamlit as st
import base64
import traceback
import textwrap
import csv


from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

import Config

#CHANGES :
# LINE 102 : ADDED LINE TO ADD URL TO CHUNK

def insert_logo(logo_path):
    if os.path.isfile(logo_path):
        with open(logo_path, "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{logo_base64}" style="max-width:250px;">', 
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.text("Logo not found!")


def remove_cache_directory(cache_directory):
    try:
        if os.path.exists(cache_directory):
            shutil.rmtree(cache_directory)
    except Exception as e:
        print(f"Error while removing cache directory: {e}")


def remove_data_directory(data_directory):
    try:
        if os.path.exists(data_directory):
            shutil.rmtree(data_directory)
    except Exception as e:
        print(f"Error while removing data directory: {e}")


def remove_persist_directory(persist_directory):
    try:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
    except Exception as e:
        print(f"Error while removing persist directory: {e}")


def clear_conversation():
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


def extract_all_urls_from_base_url(base_url, url_file_name):
    try:
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        a_tags = soup.find_all('a')
        with open(url_file_name, 'w') as f:
            for tag in a_tags:
                href = tag.get('href')
                if href is not None:  # avoid NoneType objects if any
                    if href.startswith("http"):  
                        f.write(href + '\n')
                    else:
                        f.write(f"{base_url.rstrip('/')}/{href.lstrip('/')}\n")
    except Exception as e:
        print(f"Error while extracting urls from {base_url}: {e}")

#def clean_data(paragraphs,width=80,indent=4):
    # Format each paragraph
    """ formatted_paragraphs = []
    for paragraph in paragraphs:

        # Wrap the paragraph to a specified width and add indentation
        wrapped_paragraph = textwrap.fill(
            paragraph, 
            width=width, 
            initial_indent=' ' * indent, 
            subsequent_indent=' ' * indent)

        # Add the formatted paragraph to the list
        formatted_paragraphs.append(wrapped_paragraph)
    return formatted_paragraphs """

def append_url_to_text_file(url,file_name,counter):
    
    if not os.path.exists(file_name):
        with open(file_name,"w",newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow([f"document_{counter}.txt", url])
            
    else:
        with open(file_name,"a",newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow([f"document_{counter}.txt", url])
    print("URL has been appended to file")
          
    """ if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    shutil.move(output_file, f'{data_directory}/{output_file}') """


def create_text_file_from_url(url_file_name, data_directory):
    try:
        counter = 0
        with open(url_file_name, 'r') as f:
            for url in f.readlines():
                url = url.strip()
                if url:
                    counter+=1
                    output_file = f"document_{counter}.txt"
                    response = requests.get(url)
                    #get each line and remove trailing spaces
                    if response.status_code == 200:
                        webpage_content = response.content
                        soup = BeautifulSoup(webpage_content, "html.parser")
                        paragraphs = soup.find_all('p')
                        with open(output_file, "w") as file:
                            
                            #file.write(url)
                            for paragraph in paragraphs:
                                file.write(paragraph.get_text())
                        if not os.path.exists(data_directory):
                            os.makedirs(data_directory)
                        shutil.move(output_file, f'{data_directory}/{output_file}')
                        print(f'Text has been written to {output_file}')
                        append_url_to_text_file(url=url,file_name="url_documents.csv",counter=counter)
                        
                    else:
                        print(f'Failed to retrieve webpage. Status code: {response.status_code}')      
    except Exception as e:
        traceback.print_exc()
        print(f"Error while creating text file from {url_file_name}: {e}")


    
    
def create_embeddings(data_directory, persist_directory):
    try:
        process_start_time = datetime.now()
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        documents = load_documents(data_directory)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
        #include the link to the article in the chunk
        texts = text_splitter.split_documents(documents=documents)
        
        embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        
        db = Chroma.from_documents(texts, embeddings,persist_directory=persist_directory)
        
        db.persist()
        db = None
        process_end_time = datetime.now()
        process_run_time = process_end_time - process_start_time
        run_time_seconds = round(process_run_time.total_seconds())
        print('process_start_time: ', process_start_time, '| process_end_time: ', process_end_time)
        print('process_run_time: ', run_time_seconds, ' seconds')
        print(f"created vectorstores and persisted to {persist_directory}")
    except Exception as e:
        traceback.print_exc()
        print(f"Error while creating embeddings: {e}")


def get_response(persist_directory, query):
    try:
        valid_conter = 0
        selected_page_content_list = []
        embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        stored_vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = stored_vector_db.as_retriever(search_kwargs={"k": 5})
        llm = OpenAI(openai_api_key=Config.OPENAI_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=True)
        response = qa_chain(query)
        answer, source_documents = response['result'], response['source_documents']
        chat_response = f"\nAI Assistant: {answer}\n"
        for counter, document in enumerate(source_documents, start=1):
            selected_page_content = document.page_content[:100]
            if selected_page_content not in selected_page_content_list:
                valid_conter += 1
                selected_page_content_list.append(selected_page_content)
                chat_response += f"\n> {document.metadata['source']} #{valid_conter}:\n"
                chat_response += document.page_content + '\n'
        if("i don't know" in chat_response.lower()):
            chat_response+=f"\n Please provide more context so that I may help you better." 
            #formatted string didn't work !! only in console, not in the message
        return chat_response
    except Exception as e:
        print(f"Error while getting response: {e}")



def load_data(base_url, url_file_name, data_directory):
    try:
        extract_all_urls_from_base_url(base_url, url_file_name)
        create_text_file_from_url(url_file_name, data_directory)
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error while loading data: {e}")


    


def train_model(data_directory, persist_directory, cache_directory):
    try:
        remove_persist_directory(persist_directory)
        create_embeddings(data_directory, persist_directory)
    except Exception as e:
        print(f"Error while training model: {e}")


def reset_model(data_directory, persist_directory, cache_directory):
    try:
        remove_data_directory(data_directory)
        remove_persist_directory(persist_directory)
    except Exception as e:
        print(f"Error while resetting model: {e}")


def load_single_document(file_path: str) -> Document:
    try:
        ext = "." + file_path.rsplit(".", 1)[-1]
        LOADER_MAPPING = {
            ".txt": (TextLoader, {"encoding": "utf8"}),
        }
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()[0]
        raise ValueError(f"Unsupported file extension '{ext}'")
    except Exception as e:
        print(f"Error while loading single document: {e}")


def load_documents(data_directory: str) -> List[Document]:
    LOADER_MAPPING = {
        ".txt": (TextLoader, {"encoding": "utf8"}),
    }
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(data_directory, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]

#add a separate function to clean the text. (1st priority)
#add a separate way to represent data in a form of a table. 

#pdf file : if the filename is too large then write a function to update it
#take name, first 25 characters, replace space with hyphen or underscore. 
#try to diagnose why it is not working for more than one accuracy.