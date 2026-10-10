from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters


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