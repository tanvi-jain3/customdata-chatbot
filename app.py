import streamlit as st
from streamlit_chat import message
import Config
from Helper_Module import *


base_url = Config.BASE_URL
url_file_name = Config.URL_FILE_NAME
data_directory = Config.DATA_DIRECTORY
persist_directory = Config.PERSIST_DIRECTORY
cache_directory = Config.CACHE_DIRECTORY
hide_streamlit_style = Config.hide_streamlit_style

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

insert_logo("logo.png")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.sidebar.title("AI Assistant")
base_url = st.sidebar.text_input(f'Enter website base url', f'{base_url}')

load_data_button = st.sidebar.button("Load Data", key="load_data")
train_model_button = st.sidebar.button("Train Model", key="train_model")
reset_model_button = st.sidebar.button("Reset Model", key="reset_model")
clear_conversation_button = st.sidebar.button("Clear Conversation", key="clear_conversation")


if clear_conversation_button:
    st.text("Clearing conversation...")
    clear_conversation()
    st.text("Conversation cleared successfully!")

if reset_model_button:
    st.text("Resetting model...")
    st.text("This may take a while...")
    reset_model(data_directory, persist_directory, cache_directory)
    st.text("Model reset successfully!")

if train_model_button:
    st.text("Training model...")
    st.text("This may take a while...")
    train_model(data_directory, persist_directory, cache_directory)
    st.text("Model trained successfully!")

if load_data_button:
    st.text("Loading data...")
    st.text("This may take a while...")
    load_data(base_url, url_file_name, data_directory)
    st.text("Data loaded successfully!")
    

response_container = st.container()
container = st.container()


with container:
    with st.form(key='my_form', clear_on_submit=True):
        query = st.text_area("User:", key='input', height=50)
        submit_button = st.form_submit_button(label='Send')
        
        

    if submit_button and query:
        st.text(f"Please wait, this may take a while...")
        st.session_state['messages'].append({"role": "user", "content": query})
        response = get_response(persist_directory, query)
        st.session_state['messages'].append({"role": "assistant", "content": response})
        print(response)
        st.session_state['past'].append(query)
        st.session_state['generated'].append(response)


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',avatar_style="identicon",seed="Salem")
            #avatar style from https://www.dicebear.com/styles/identicon (only supported source). 
            #logo upload is supported in the latest version of the package
            message(st.session_state["generated"][i], key=str(i),avatar_style="identicon",seed="Mittens")


