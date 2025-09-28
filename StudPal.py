from transformers import pipeline
import gradio as gr
import torch
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.dataclasses import ChatMessage
import PyPDF2
from haystack.components.generators import BaseGenerator

# --- PDF Extraction ---
def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.
    Args:
        file_path (str): The path to the PDF file.
    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        return f"Error reading PDF file: {e}"
    return text if text else "No text found in the PDF."

# --- Haystack RAG Setup ---
# Initialize document store and add extracted PDF text as a document
document_store = InMemoryDocumentStore()
document_store.write_documents([Document(content=extract_text_from_pdf("Manuel_de_Maths_es.pdf"))])

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

# --- Hugging Face Model Setup ---
task = "text2text-generation"
model = "google/flan-t5-base"

# Translation pipelines for French <-> English
en_to_fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
fr_to_en = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")

# Main text generation pipeline (with temperature and device settings)
generator = pipeline(
    task,
    model,
    torch_dtype=torch.float16,  # Use float16 for better performance on compatible hardware
    device=0,                   # Use GPU if available
    temperature=0.7,            # Adjust temperature for creativity and precision
    trust_remote_code=True      # Trust remote code for model compatibility
)

# --- Custom Haystack Generator Component ---
class HFPipelineGenerator(BaseGenerator):
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
llm = HFPipelineGenerator(generator)

# --- Build Haystack RAG Pipeline ---
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm.messages")

# --- Text Generation with Translation ---
def translate_n_generate_text(prompt):
    """
    Translate the input prompt from French to English, generate a response in English using RAG,
    and translate it back to French.
    Args:
        prompt (str): The input text prompt in French.
    Returns:
        str: The generated response translated back to French.
    """
    if not prompt:
        return "Please enter a valid prompt."

    # Translate question from French to English
    question_in_en = fr_to_en(
        prompt.strip(),
        max_length=100,
        num_return_sequences=1,
        repetition_penalty=1.3
    )[0]['translation_text']

    # Use RAG pipeline to get a context-aware answer in English
    results = rag_pipeline.run(
        {
            "retriever": {"query": question_in_en.strip()},
            "prompt_builder": {"question": question_in_en.strip()},
        }
    )

    # Get the generated answer in English
    answer_en = results['llm']['replies'][0]

    # Translate answer back to French
    answer_fr = en_to_fr(
        answer_en.strip(),
        num_return_sequences=1,
        repetition_penalty=1.3
    )[0]['translation_text']

    return answer_fr

# --- Gradio Interface ---
root = gr.Interface(
    fn=translate_n_generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs=gr.Textbox(label="Your answer!"),
    title="StudPal AI",
    description="An AI-powered study companion that helps you with your studies by generating text based on your prompts."
)

# --- Launch Gradio App ---
root.launch()

