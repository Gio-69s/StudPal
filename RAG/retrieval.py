# Importation du module d'ingestion pour réutiliser la base vectorielle déjà créée.
import ingestion

# Création du retriever FAISS.
# search_type="similarity" signifie que la recherche se fait par similarité sémantique.
# search_kwargs={"k": 4} indique qu'on veut récupérer les 4 chunks les plus proches.
retriever = ingestion.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Question posée par l'utilisateur.
# Exemple : recherche de contenu sur la factorisation pour calculer une limite.
question = "Comment calculer une limite en utilisant la factorisation ?"

# Exécution de la recherche dans la base vectorielle.
# On recherche les passages les plus pertinents par rapport à la question.
docs = retriever.invoke(question)

# Affichage du contenu des documents retrouvés.
# Chaque document correspond à un chunk du PDF jugé pertinent.
for doc in docs:
    print(f"{doc.page_content}\n------------")