from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators import HuggingFaceLocalGenerator


# --- Haystack RAG Setup ---
# Initialize document store and add extracted PDF text as a document
document_store = InMemoryDocumentStore()
document_store.write_documents()

# Set up BM25 retriever for document search
retriever = InMemoryBM25Retriever(document_store=document_store)

# Prompt template for RAG (Jinja-style for Haystack)
prompt_template = (
    "Use the following context to answer the question.\n\n"
    "Context:\n"
    "{% for doc in documents %}{{ doc.content }}\n{% endfor %}\n"
    "Question: {{question}}\n"
    "Answer:"
)
prompt_builder = PromptBuilder(template=prompt_template)

# --- Custom Haystack Generator Component ---
class HFPipelineGenerator(HuggingFaceLocalGenerator):
    """
    Custom Haystack generator that wraps a Hugging Face pipeline.
    """
    def __init__(self, hf_pipeline):
        super().__init__()
        self.hf_pipeline = hf_pipeline

    def run(self, messages, **kwargs):
        # Combine all message contents into a single prompt
        prompt = "\n".join([msg.content for msg in messages])
        response = self.hf_pipeline(prompt, max_length=50, num_return_sequences=1)
        return {"replies": [response[0]['generated_text']]}

# Instantiate the custom generator for RAG
llm = HFPipelineGenerator()

# --- Build Haystack RAG Pipeline ---
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm.messages")


