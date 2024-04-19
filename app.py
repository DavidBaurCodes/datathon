# pip install langchain streamlit langchain-openai python-dotenv
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
import os
import toml


# load api keys region etc from secrets.toml file
os.environ["OPENAI_API_KEY"] = st.secrets['openai_api_key']
pinecone_api_key = st.secrets['pinecone_api_key']
pinecone_region = st.secrets['pinecone_region']
pinecone_index_name = st.secrets['pinecone_index_name']

st.set_page_config(page_title="RAG - TEST", page_icon="🔗", layout="wide")

st.title("Datathon - SOP RAG")

# Initialize chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
embeddings = OpenAIEmbeddings(model="text-embedding-3-large") #Initialisiere die OpenAI-Embeddings
vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)  # Initialisiere den Pinecone-Vektor-Speicher    


def get_context_retriever_chain(vectorstore):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()  # Initialisiere den Retriever

    search_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", '''Basierend auf dem vorherigen Gespräch, generiere eine Suchanfrage, 
         um relevante Informationen für die Konversation zu erhalten. Gehe auf neue Fragen ein
         Ziehe neue Quellen heran, um die Antwort zu generieren.''')
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, search_prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''Deine Rolle ist die eines SOP-Assistenten am Universitätsklinikum Leipzig (UKL), 
        spezialisiert auf die Standard Operating Procedures (SOPs) des Klinikums. Du verfügst über eine umfassende 
        Kenntnis dieser SOPs und nutzt diese, um präzise und strukturierte Antworten in medizinischer Fachsprache zu 
        liefern. Als zentrale Anlaufstelle für das medizinische Fachpersonal - einschließlich Ärztinnen, Ärzte, Pflegekräfte 
        und weitere Gesundheitsberufe - ist es deine Aufgabe, auf deren spezifische Fragen zu den SOPs einzugehen.
        Deine Antworten sollten sich stets auf den vorgegebenen Kontext beziehen. Sollten Unklarheiten bezüglich der Fragen bestehen, zögere nicht, Rückfragen zu stellen, um die Anfrage effektiv zu adressieren. Es ist wichtig, dass du in deinen Ausführungen die relevanten Teile der SOPs zitierst, um Transparenz und die Gültigkeit deiner Antworten zu gewährleisten.
        Dein Kommunikationsstil sollte freundlich und professionell sein, passend zum Umgang mit Fachpersonal. Bei Unsicherheiten oder fehlenden 
        Informationen ist es essentiell, ehrlich zu kommunizieren und zu klären, dass weitere Informationen benötigt werden, 
        um eine fundierte Antwort geben zu können. Dein oberstes Gebot ist, ausschließlich Informationen zu nutzen, 
        die direkt aus den SOPs, also deinem Context stammen. Gib immer die Namen der SOPs als Quelle am Ende deiner Ausgabe an (falls
        sinnvoll alle die du verwendet hast).
        Kontext: {context}'''),  
        # ("system", '''Deine Rolle ist SOP Assistent. Du hast ausführlichen Kontext über SOPS im Universitätsklinikum Leipzig
        #  (UKL) und hilfst Ärzten und Ärztinnen, sowie Pflegepersonal Fragen zu den SOPs zu beantworten. 
        #  Sei ausführlich in deinen Angaben und versuche möglichst präzise auf die dir gestelle Frage zu antworten. Formuliere Antowrten mit 
        #  medizinischer Sprache. Sollten dir innerhalb deines Kontexts keine Antworten möglich sein gib dies ehrlich an
        #  und bitte halte dich streng an den Dir vorgegebenen Kontext:\n\n{context}"). Bitte gib falls dir vorliegend immer die 
        #  SOPs namentlich am ende an die du verwendet hast um zu deiner Antwort zu kommen.'''),  
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

 

# get_response definiert einen Generator, der die Antworten streamt
def get_response(query, chat_history):
    # Initialisiere deine Chains
    retriever_chain = get_context_retriever_chain(vectorstore)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    # Bereite deine Eingabedaten vor
    input_data = {
        "chat_history": chat_history,
        "input": query
    }
    
    # Direkter Aufruf von conversation_rag_chain.stream mit input_data
    response_stream = conversation_rag_chain.stream(input_data)
    
    # Iteriere durch den Stream und yield nur die 'answer' Teile
    for response in response_stream:
        if 'answer' in response:
            yield response['answer']

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

#user input
user_query = st.chat_input("Deine Frage")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Direkte Nutzung von st.write_stream mit dem Generator
    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(ai_response))   
