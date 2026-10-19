# Importation du module d'ingestion pour réutiliser la base vectorielle déjà créée.
import ingestion
import prompts

# Création du retriever FAISS.
# search_type="similarity" signifie que la recherche se fait par similarité sémantique.
# search_kwargs={"k": 4} indique qu'on veut récupérer les 4 chunks les plus proches.
retriever = ingestion.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# Exécution de la recherche dans la base vectorielle.
# On recherche les passages les plus pertinents par rapport à la question.
docs = retriever.invoke(prompts.question)

# Liste qui contiendra le texte de chaque document récupéré.
chunk_text = []

# Parcourt les documents trouvés et ajoute leur contenu textuel à la liste.
for doc in docs :
   chunk_text.append(doc.page_content)
