# Importation des bibliothèques nécessaires pour :
# 1. charger le PDF,
# 2. découper le document en morceaux cohérents,
# 3. générer des embeddings vectoriels,
# 4. stocker puis exploiter la base FAISS.
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Chemin du PDF à indexer.
# Ce fichier contient les pages à transformer en documents exploitables pour la recherche sémantique.
loader = PyMuPDFLoader(
    r"C:\Users\giova\OneDrive\Documents\GitHub\StudPal\datas\Maths\Dossier_Af\Mathematiques Resumé StudPal.pdf"
)

# Lecture du PDF et conversion en liste de documents LangChain.
documents = loader.load()


# Configuration du découpage du texte.
# chunk_size = nombre maximal de caractères par morceau.
# chunk_overlap = nombre de caractères à chevaucher entre deux morceaux.
# Cela permet de conserver du contexte entre les sections et d'éviter de casser la logique du texte.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)


# Découpe chaque document en plusieurs chunks plus petits.
# Ces morceaux seront ensuite intégrés dans la base vectorielle pour la recherche.
chunks = splitter.split_documents(documents)


# Création du modèle d'embeddings multilingue.
# Il transforme chaque chunk en vecteur numérique que FAISS pourra comparer rapidement.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# Génération de la base vectorielle FAISS.
# Chaque chunk est associé à son embedding, ce qui permet ensuite des recherches par similarité.
vector_store = FAISS.from_documents(chunks, embeddings)

# Sauvegarde locale de la base pour la réutiliser sans recalculer tout l'index à chaque lancement.
vector_store.save_local("my_vector_store")


# Fonction principale de test rapide.
# Elle vérifie simplement que les objets essentiels ont bien été créés.
def main():
    if documents:
        print(True)
    if chunks:
        print(True)
    if vector_store:
        print(True)


# Point d'entrée du script.
# La fonction main ne s'exécute que si le fichier est lancé directement.
if __name__ == "__main__":
    main()