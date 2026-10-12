import ingestion
import prompts

retriever = ingestion.vector_store.as_retriever(
    search_type ="similarity",
    search_kwargs ={"k":4}
)

docs =retriever.invoke(prompts.question)