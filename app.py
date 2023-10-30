# Import required libraries
import os
from dotenv import load_dotenv
from itertools import zip_longest
from PIL import Image
import requests
import streamlit as st
from streamlit_chat import message
from streamlit_lottie import st_lottie
from langchain.chat_models import ChatOpenAI
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)




st.set_page_config(page_title = 'Chat_PDF' , layout= "wide")
z1, z2 = st.columns([30,20])
with z1:
    new_title = '<p style="font-family:verdana; color:black; font-size: 50px; float:right;">Chat_PDF</p>'
    st.markdown("###")
    st.markdown(new_title, unsafe_allow_html=True)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_filter = load_lottieurl('https://lottie.host/a2b66809-ad4f-4f74-96b0-a9dcaecc2ade/0ZabMBKLDn.json')

with z2:
    st_lottie(
        lottie_filter,
        speed=2,
        reverse=False,
        loop=True,
        quality="medium", # medium ; high
        height=80,
        width=80,
        key=None,)

selected = option_menu(menu_title=  None, 
                            options = ['General_Chat' , 'Chat_With_PDF' ],
                            icons= ['graph-up' , 'reception-4'],
                            orientation= "horizontal", )

with st.sidebar:
    if 'OPENAI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['OPENAI_API_KEY']
    else:
        replicate_api = st.text_input('API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['OPENAI_API_KEY'] = replicate_api

if selected == "Chat_With_PDF":
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    def get_vectorstore(text_chunks):
        embeddings = OpenAIEmbeddings()
        # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    

    def get_conversation_chain(vectorstore):
        llm = ChatOpenAI()
        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    
    def handle_userinput(user_question):
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                z1, z2 = st.columns([50,50])
                with z1:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                z1, z2 = st.columns([70,30])
                with z2:
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)



    def main():
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        st.header("Chat with multiple PDFs :books:")
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
                    

    if __name__ == '__main__':
        main()


else:
    # Initialize session state variables

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []  # Store AI generated responses

    if 'past' not in st.session_state:
        st.session_state['past'] = []  # Store past user inputs

    if 'entered_prompt' not in st.session_state:
        st.session_state['entered_prompt'] = ""  # Store the latest user input

    # Initialize the ChatOpenAI model
    chat = ChatOpenAI(
        temperature=0.5,
        model_name="gpt-3.5-turbo"
    )



    def build_message_list():
        """
        Build a list of messages including system, human and AI messages.
        """
        # Start zipped_messages with the SystemMessage
        zipped_messages = [SystemMessage(
            content="You are a helpful AI assistant talking with a human. If you do not know an answer, just say 'sorry, I May not be able to assist you with this.', do not make up an answer.")]

        # Zip together the past and generated messages
        for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
            if human_msg is not None:
                zipped_messages.append(HumanMessage(
                    content=human_msg))  # Add user messages
            if ai_msg is not None:
                zipped_messages.append(
                    AIMessage(content=ai_msg))  # Add AI messages

        return zipped_messages


    def generate_response():
        """
        Generate AI response using the ChatOpenAI model.
        """
        # Build the list of messages
        zipped_messages = build_message_list()

        # Generate response using the chat model
        ai_response = chat(zipped_messages)

        return ai_response.content


    # Define function to submit user input
    def submit():
        # Set entered_prompt to the current value of prompt_input
        st.session_state.entered_prompt = st.session_state.prompt_input
        # Clear prompt_input
        st.session_state.prompt_input = ""


    # Create a text input for user
    st.text_input('YOU: ', key='prompt_input', on_change=submit)


    if st.session_state.entered_prompt != "":
        # Get user query
        user_query = st.session_state.entered_prompt

        # Append user query to past queries
        st.session_state.past.append(user_query)

        # Generate response
        output = generate_response()

        # Append AI response to generated responses
        st.session_state.generated.append(output)

    # Display the chat history
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            # Display AI response
            message(st.session_state["generated"][i], key=str(i))
            # Display user message
            message(st.session_state['past'][i],
                    is_user=True, key=str(i) + '_user')