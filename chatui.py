import streamlit as st
import json
import os
import requests
import chromadb
from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_HOST = os.getenv("CHROMA_HOST")

CHROMA_PORT = 8000
chroma_client = chromadb.Client(Settings(chroma_api_impl="rest",
                                         chroma_server_host=CHROMA_HOST,
                                         chroma_server_http_port=CHROMA_PORT
                                         ))
chroma_collection = "sinhalaTestProd"
collection = chroma_client.get_or_create_collection(
    name=chroma_collection, embedding_function=OpenAIEmbeddings())

file_path = "./sample_files/SinhalaPanthiya.pdf"

# Load the pdf file
loader = PyPDFLoader(file_path)
document = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0)
chunked_documents = text_splitter.split_documents(document)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
    collection_name=chroma_collection,
    client=chroma_client,
)
retriever = vectordb.as_retriever()

sinhalaPanthiya_system_prompt = """
        You’re a helpful self service agent developed by Sinhala Panthiya to help students that want to sign up to classes and courses provided by Sinhala Panthiya.\
        Your goal is to provide students/parents with information on the available classes, and direct them to the website  sinhalapanthiya.lk \
        when they are ready to register for classes. 
"""
contexualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sinhalaPanthiya_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
llm = ChatOpenAI(model_name="gpt-4o",
                 api_key=OPENAI_API_KEY, temperature=0)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contexualize_q_prompt
)
sinhalaPanthiya_qa_system_prompt = """
        Instructions:
                - ⁠Politely the greet the user; example; “Hi! Welcome to Sinhala Panthiya. How can I help you today?”\
                - Ask clarifying questions if the user's request is ambiguous.\
                - Keep your responses brief and within 1-3 sentences. Your responses are meant to mimic SMS conversations, not long-form explanations.\
                - You can speak multiple languages. Reply in the user’s language.\
                - Use bullet points or numbered lists where appropriate.\
                - Don't make up, promise or create anything that's not explicitly given here.\
                - Stay on topic and ensure your responses are relevant to the user's query.\
                - If the user's question is not covered, or is spam like messages or is not relevant, don't answer it. Instead ask them to contact the business directly.\
                {context}
   """
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sinhalaPanthiya_qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)
store = {}


def get_session_history(user_id: str, conversation_id: str,) -> BaseChatMessageHistory:
    global store
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        )
    ],
)


def chatbot_response(phone_number, user_input):
    answer = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"user_id": phone_number, "conversation_id": phone_number}
        },
    )
    response = answer['answer']
    return response


def chat_app(phone_number, user_input, history):
    if phone_number not in history:
        history[phone_number] = []
    history[phone_number].append(("User", user_input))
    bot_response = chatbot_response(phone_number, user_input)
    history[phone_number].append(("Bot", bot_response))
    chat_history = [f"{sender}: {message}" for sender,
                    message in history[phone_number]]
    return chat_history, history


def main():
    st.title("Chatbot App")
    phone_number = st.text_input("Phone Number")
    user_input = st.text_input("Your Message")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}
    if st.button("Send"):
        chat_history, st.session_state["chat_history"] = chat_app(
            phone_number, user_input, st.session_state["chat_history"])
        st.text_area("Chat History", value="\n".join(chat_history), height=300)


if __name__ == "__main__":
    main()
