from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from  dotenv import load_dotenv
import os

# --- Load the PDF -------------------------------------------------------------
# Replace this path with the course PDF you want to ingest.
loader = PyPDFLoader(
    r"C:\Users\giova\OneDrive\Documents\GitHub\StudPal\datas\Maths\Dossier_Af\analyse_fonctions.pdf"
)

# Load the PDF into a list of Document objects (one per page by default).
docs = loader.load()
print(f"Loaded {len(docs)} document pages from PDF")

# --- Split the text into smaller chunks ---------------------------------------
# Chunking helps create more fine-grained embeddings and improves retrieval quality.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # target size per chunk
    chunk_overlap=50,  # overlap between chunks to preserve context
)
chunked_docs = splitter.split_documents(docs)
print(f"Created {len(chunked_docs)} text chunks")

# --- Embed the chunks ---------------------------------------------------------
# Use a small, fast sentence-transformer model via HuggingFace.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- Build and save the FAISS vector store ------------------------------------
# FAISS stores the embeddings for fast similarity search.
vector_store = FAISS.from_documents(chunked_docs, embeddings)

# Persist the vector store locally for later reuse.
vector_store.save_local("my_vector_store")
print("Vector store saved to 'my_vector_store'")

# --- Simple retrieval example --------------------------------------------------
# Create a retriever and query it for relevant chunks.
retriever = vector_store.as_retriever(search_type= "similarity",search_kwargs={"k": 4})

# Load environment variables from a .env file (for API keys, endpoint URLs, etc.).
load_dotenv()

# Initialize the LLM pipeline (uses a HuggingFace model under the hood).
# Adjust `model_kwargs` as needed for temperature, token limits, etc.
llm = HuggingFacePipeline(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={
        "temperature": 0.3,
        "max_token": 502,
    } ,
)


# --- Prompt templates --------------------------------------------------------
# Template for a question-answering flow. The {context} placeholder is filled with
# the chunks retrieved from the vector store (relevant text from the PDF).
qa_template = '''Tu es un assistant d'études précieux pour un étudiant de BAC ivoirien. 
Utilises le contexte suivant pour répondre à la question de manière claire et précise. 
Si tu ne sais pas, dis-le - ne ments surtout pas.

Context :
{context}

Question : {question}

Réponse : '''

# Template for generating a full exercise + correction based on a given context.
exercise_template = '''Tu es un professeur de mathématiques en terminale scientifique (niveau bac+2). 
À partir du programme suivant, élabores un exercice de type bac+2 avec sa correction complète à la fin.
Contenu du programme:
{context}

Génération de l'exercice maintenant :'''

# Create PromptTemplate objects for each use case.
qa_prompt = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question"],
)

exercise_prompt = PromptTemplate(
    template=exercise_template,
    input_variables=["context"],
)






