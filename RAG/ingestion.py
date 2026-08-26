from importlib import import_module


_document_loaders = import_module("langchain_community.document_loaders")
DirectoryLoader = _document_loaders.DirectoryLoader
PyMuPDFLoader = _document_loaders.PyMuPDFLoader
_text_splitter = import_module("langchain_text_splitters")
RecursiveCharacterTextSplitter = _text_splitter.RecursiveCharacterTextSplitter
_embeddings = import_module("langchain_huggingface")
HuggingFaceEmbeddings = _embeddings.HuggingFaceEmbeddings
vector_store = import_module("langchain_community.vectorstores")
FAISS = vector_store.FAISS



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