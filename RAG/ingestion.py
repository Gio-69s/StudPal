# Importation des bibliothèques nécessaires pour charger le PDF,
# découper le texte et créer les embeddings vectoriels.
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Charge le fichier PDF contenant les documents à indexer.
loader = PyMuPDFLoader(r"C:\Users\giova\OneDrive\Documents\GitHub\StudPal\datas\Maths\Dossier_Af\Mathematiques Resumé StudPal.pdf")
documents = loader.load()
if documents :
    print(True)

# Découpe les documents en petits passages avec un chevauchement pour préserver le contexte
# entre les morceaux et éviter de perdre des informations lors de la recherche.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)

# Divise les documents chargés en plusieurs morceaux plus petits.
chunks = splitter.split_documents(documents)
if chunks :
    print(True)

# Transforme chaque passage en vecteur avec un modèle multilingue.
# Cela permet ensuite d'effectuer une recherche sémantique dans les documents.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Crée la base vectorielle FAISS à partir des chunks et des embeddings.
# Ensuite, elle est sauvegardée localement pour être réutilisée plus tard.
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("my_vector_store")
if vector_store : 
    print(True)