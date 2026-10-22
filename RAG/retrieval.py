# Importation du module d'ingestion pour réutiliser la base vectorielle déjà créée.
import ingestion

# Création du retriever FAISS.
# search_type="similarity" signifie que la recherche se fait par similarité sémantique.
# search_kwargs={"k": 4} indique qu'on veut récupérer les 4 chunks les plus proches.
retriever = ingestion.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Recherche les extraits les plus pertinents dans la base vectorielle pour la question reçue.
def retrieving_and_context_chunk(question_received):
   
    # Interroge le retriever et récupère les documents correspondants.
    docs = retriever.invoke(question_received)
    # Extrait le contenu textuel de chaque document trouvé.
    chunk_text = [doc.page_content for doc in docs]
    # Renvoie la liste des extraits afin qu'ils puissent être intégrés au prompt.
    return chunk_text
