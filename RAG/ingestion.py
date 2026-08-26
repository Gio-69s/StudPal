from importlib import import_module


_document_loaders = import_module("langchain_community.document_loaders")
DirectoryLoader = _document_loaders.DirectoryLoader
PyMuPDFLoader = _document_loaders.PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = DirectoryLoader(
    "datas/Maths",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("my_vector_store")