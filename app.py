import streamlit as st 
import openai 
from langchain_openai import ChatOpenAI 
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
# from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import os 
import time
import bs4
import uuid
from dotenv import load_dotenv
load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


## RAG ChatBot from PDF Loading 


##############  Prompt Template ############ 

sys_prompt = """
                You are an intelligent, concise, and helpful QnA assistant. Answer the user's questions clearly and accurately.

                - If the question is ambiguous, ask for clarification.
                - If you do not know the answer, say “I am not sure about that” instead of making up an answer.
                - Keep your responses brief but informative unless the user requests a detailed explanation.
                - Use simple language that is easy to understand.
                """


# rag_prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#     You are a concise, helpful assistant. Answer the user's question based **only on the given context**. 

#     Guidelines:
#     - Reference specific details from the context when answering.
#     - If the answer is not in the context, say: 'I couldn't find the answer in the provided context.'
#     - Keep your answers clear and concise.

#     Context:
#     {context}
#     """),
#     ("user", "{input}")
# ])

rag_prompt = (
    """
    You are a concise, helpful assistant. Answer the user's question based **only on the given context**. 

    Guidelines:
    - Reference specific details from the context when answering.
    - If the answer is not in the context, say: 'I couldn't find the answer in the provided context.'
    - Keep your answers clear and concise.

    Context:
    {context}
    """
)


########## ChatBot History Management ###########

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)


contextual_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}") 
    ]
)

def create_session_config():
    return {"configurable": {"session_id": str(uuid.uuid4())}}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


def create_vectors_embeddings(docs, embedding_type, document_type):

    """
    args:
        - pdf_file: path to file of pdf: str  
        - embedding type: str
        - Document Type ["pdf", "web"]
    """

    if "vectors" not in st.session_state:

        ## Different Types of Embeddings 

        if st.session_state.embedding_type == "OpenAI":
            st.session_state.embeddings = OpenAIEmbeddings()
        elif st.session_state.embedding_type == "Ollama":
            st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        elif st.session_state.embedding_type == "HuggingFace":
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        ## Different Types of Loaders

        if document_type == "pdf":
            st.session_state.loader = PyPDFLoader(docs)  ## Data Ingestion Step 
        elif document_type == "web":
            st.session_state.loader = WebBaseLoader(web_path=docs)  ## Data Ingestion Step 

        
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents= st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Embeddings created and stored successfully!")

def create_llm(model_type, llm_name):
    match model_type:
        case "GPT Models":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            st.session_state.llm = ChatOpenAI(
                model=llm_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p)
        
        case "Groq Models":
            st.session_state.llm = ChatGroq(
                model=llm_name,
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
        
        case "Ollama Models":
            st.session_state.llm = ChatOllama(
                model=llm_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )

def generate_response(question, system_prompt, model_type, llm_name, temperature, max_tokens, top_p):

    """
    args: 
        - question 
        - system_prompt
        - model_type 
        - llm_name 
        - temperature 
        - max_tokens 
        - top_p
        - retriever
        
    """

    # Default LLM is OPENAI
    llm=ChatOpenAI(
                model=llm_name,
                temperature=temperature,
                max_tokens=max_tokens)

    match model_type:
        case "GPT Models":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            llm = ChatOpenAI(
                model=llm_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p)
        
        case "Groq Models":
            llm = ChatGroq(
                model=llm_name,
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
        
        case "Ollama Models":
            llm = ChatOllama(
                model=llm_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
    prompt = ChatPromptTemplate([
        ("system", system_prompt if system_prompt is not None else sys_prompt),
        ("user", "{input}")
    ])

    parser = StrOutputParser()

    chain = prompt | llm | parser
    # qna_chain = create_stuff_documents_chain(llm, prompt)
    # rag_chain = create_retrieval_chain(retriever, qna_chain)

    response = chain.invoke({"input", question})
    return response


## Streamlit App

st.title("QnA RAG Chatbot")



############ Web BASED RAG CHATBOT FUNCTIONS#####################

# Enter a Link in Sidebar 
st.sidebar.title("Settings")

st.sidebar.subheader("Mode of Chatbot")
st.session_state.mode_selection = st.sidebar.segmented_control(
    "Mode-Type", ["Simple Chatbot", "Web-based RAG Chatbot", "PDF RAG Chatbot"]
)

web_link = st.sidebar.text_input("Enter an URL to Load", placeholder="https://blog.langchain.com/how-to-build-an-agent/")


################# RAG CHATBOT Functions####################
    
file_data = st.sidebar.file_uploader(
    label="Upload a PDF File To RAG", 
    accept_multiple_files=False,
    type="pdf",)

st.session_state.embedding_type = st.sidebar.pills(label="Embedding Type: ", options=["OpenAI", "Ollama", "HuggingFace"])

if st.sidebar.button("Document Embeddings"):
    
    if st.session_state.mode_selection == "PDF RAG Chatbot" and file_data is not None:
        ## Save the file temporarily 
        with open("temp_doc.pdf", "wb") as f:
            f.write(file_data.read())

        create_vectors_embeddings("temp_doc.pdf", st.session_state.embedding_type, "pdf")
    
    elif st.session_state.mode_selection == "Web-based RAG Chatbot" and web_link is not None:
        create_vectors_embeddings(web_link, st.session_state.embedding_type, "web")



st.session_state.model_selection = "GPT Models" ## Default
st.session_state.model_selection = st.sidebar.segmented_control(
    "Model-Type", ["GPT Models", "Groq Models", "Ollama Models"]
)

st.session_state.llm_name = "gpt-3.5-turbo" ## Default
match st.session_state.model_selection:
    case "GPT Models":
        st.session_state.llm_name = st.sidebar.selectbox("Select An OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"])
    
    case "Groq Models":
        st.session_state.llm_name = st.sidebar.selectbox("Select An Groq TPU Model", ["deepseek-r1-distill-llama-70b", "meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3-32b", "llama-3.1-8b-instant"])

    case "Ollama Models":
        st.session_state.llm_name = st.sidebar.selectbox("Select an Ollama Model", "gemma:2b")

##Adjusters Slidebar 
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=1000, value=100)
top_p = st.sidebar.slider("Top-P",  min_value=0.0, max_value=1.0, value=0.7)

st.session_state.temperature = temperature 
st.session_state.max_tokens = max_tokens 
st.session_state.top_p = top_p 


## System Prompt in Sidebar 

st.session_state.system_prompt = st.sidebar.text_area("System Prompt", height="content", placeholder=sys_prompt)


## Main interface for user input 
st.subheader("Model Selected")
st.badge("Model Type: " + st.session_state.model_selection if st.session_state.model_selection is not None else "GPT Models", 
         icon=":material/check:", 
         color="green")
st.badge("Model Name: " + st.session_state.llm_name,  
         icon=":material/robot_2:", 
         color="blue")

st.subheader("Model Hyperparameters Settings: ")
st.badge("Temperature: " + str(st.session_state.temperature), icon=":material/settings:", color="red")
st.badge("Max Tokens: " + str(st.session_state.max_tokens), icon=":material/settings:", color="red")
st.badge("Top-P: " + str(st.session_state.top_p),  icon=":material/settings:", color="red")

if 'store' not in st.session_state:
    st.session_state.store = {}

if 'config' not in st.session_state:
    st.session_state.config = create_session_config()

st.badge("Session ID: " + st.session_state.config["configurable"]["session_id"], icon=":material/check:", color="green")


######### CHAT UI ##########

# Session state for messages 
st.subheader("Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []

## Display Chat History 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask Anything")

############ Processing the Type of Chatbot ###################

if user_input:

    # Handle User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.mode_selection == "Simple Chatbot":

        start = time.process_time()
        response = generate_response(user_input, st.session_state.system_prompt, st.session_state.model_selection, st.session_state.llm_name, st.session_state.temperature, st.session_state.max_tokens, st.session_state.top_p)
        st.write(f"Response Time: {time.process_time() - start:.3f} sec")
        # st.markdown(response['answer'])

        ## Show the AI Response in Chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


    elif st.session_state.mode_selection == "PDF RAG Chatbot" or st.session_state.mode_selection == "Web-based RAG Chatbot":


        create_llm(st.session_state.model_selection, st.session_state.llm_name)

        # document_chain = create_stuff_documents_chain(st.session_state.llm, rag_prompt)
         # retrieval_chain = create_retrieval_chain(retriever, document_chain)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", rag_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(st.session_state.llm, qa_prompt) ## For filling the Context

        retriever = st.session_state.vectors.as_retriever()

        ## History aware retriever 
        history_aware_retriever = create_history_aware_retriever(st.session_state.llm,retriever, contextual_prompt)       

        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            history_messages_key="chat_history",
            output_messages_key="answer"
        )


        start = time.process_time()
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config = st.session_state.config,
        ) 
        st.write(f"Response Time: {time.process_time() - start:.3f} sec")
        # st.write(response['answer'])

        ## With Streamlit Expander 
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("--------------------------------")

        ## Show the AI Response in Chat
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        with st.chat_message("assistant"):
            st.write(response['answer'])
