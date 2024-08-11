# from langchain.embeddings import OpenAIEmbeddings
import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client.models import VectorParams, Distance
import os
from dotenv import load_dotenv

load_dotenv()

# Load your documents
loader = TextLoader("about_me.txt")
documents = loader.load()

# Split the documents into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Embed the documents using OpenAI's embeddings
embeddings = OpenAIEmbeddings()

# Create the Qdrant vector store using connection parameters
vector_store = Qdrant.from_documents(
    documents=docs,
    embedding=embeddings,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="akilesh_docs"
)

# Set up the OpenAI LLM
llm = ChatOpenAI(model_name="gpt-4")

# Assume vector_store is defined elsewhere, and create a retriever from it
retriever = vector_store.as_retriever()

# Create the RAG chain using the retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

print(qa_chain)


def respond(message):
    # Generate the LLM response
    result = qa_chain({"query": message})
    response = result['result']
    return response


def generate_answer(message: str, history: list) -> str:
    # Get the new response from the LLM
    new_response = respond(message)
    return new_response


# Create the ChatInterface
demo = gr.ChatInterface(
    fn=generate_answer, title="RAG App | Learn More About Me!", multimodal=False, retry_btn=None, undo_btn=None, clear_btn=None)

# Launch the Gradio interface
demo.launch()
