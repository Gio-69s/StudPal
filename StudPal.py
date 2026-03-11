from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
# Note: For real applications, use the `Retriever.get_relevant_documents` API instead of
# accessing underlying private APIs like `_get_relevant_documents`.
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

relevant_chunk= retriever._get_relevant_documents(run_manager="Limite")

for chunk in relevant_chunk :
    print(chunk.page_content)
    print("---")






