from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader(r"C:\Users\giova\OneDrive\Documents\GitHub\StudPal\datas\Maths\Dossier_Af\analyse_fonctions.pdf")
docs=loader.load()
print(len(docs))

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500 ,
    chunk_overlap=50
)
chunk = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store= FAISS.from_documents(chunk, embeddings)

vector_store.save_local("my vector_store")






