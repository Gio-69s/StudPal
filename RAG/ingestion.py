from importlib import import_module


# Charge les classes LangChain dynamiquement pour limiter les imports directs.
_document_loaders = import_module("langchain_community.document_loaders")
DirectoryLoader = _document_loaders.DirectoryLoader
PyMuPDFLoader = _document_loaders.PyMuPDFLoader
_text_splitter = import_module("langchain_text_splitters")
RecursiveCharacterTextSplitter = _text_splitter.RecursiveCharacterTextSplitter
_embeddings = import_module("langchain_huggingface")
HuggingFaceEmbeddings = _embeddings.HuggingFaceEmbeddings
vector_store = import_module("langchain_community.vectorstores")
FAISS = vector_store.FAISS


# Recherche tous les fichiers PDF du dossier de ressources mathématiques.
loader = DirectoryLoader(
    "datas/Maths",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
)

documents = loader.load()

# Découpe les documents en passages avec un chevauchement pour préserver le contexte.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)

chunks = splitter.split_documents(documents)

# Transforme chaque passage en vecteur grâce à un modèle multilingue.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Construit l'index de recherche et le sauvegarde pour le retrouver à l'exécution.
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("my_vector_store")