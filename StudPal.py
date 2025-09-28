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

#Extract from PDF
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
# Example usage:
# pdf_text = extract_text_from_pdf("example.pdf")
#print("PDF text extraction function is ready.")

#Setting up the document store and retriever
document_store=InMemoryDocumentStore()
document_store.write_documents=([Document(content=extract_text_from_pdf("Manuel_de_Maths_es.pdf"))])
retriever=InMemoryBM25Retriever(document_store=document_store)

# Setting up the prompt builder
prompt_template = (
    "Use the following context to answer the question.\n\n"
    "Context:\n"
    "{% for doc in documents %}{{ doc.content }}\n{% endfor %}\n"
    "Question: {{question}}\n" 
    "Answer:"
)
prompt_builder = PromptBuilder(template=prompt_template)

# Initialize the text generation pipeline with a specific task and model
task = "text2text-generation"
model = "google/flan-t5-base"



#Translation pipelines
en_to_fr =pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
fr_to_en = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")

# Loading the model with a specified temperature for generation
generator = pipeline(task,
                    model,
                    torch_dtype=torch.float16,  # Use float16 for better performance on compatible hardware
                    device=0,
                    temperature=0.7, # Adjust temperature for creativity and precision in responses
                    trust_remote_code=True, # Trust remote code for model compatibility
                    )
# Custom generator class to integrate Hugging Face pipeline with Haystack
class HFPipelineGenerator(BaseGenerator):
    def __init__(self, hf_pipeline):
        super().__init__()
        self.hf_pipeline = hf_pipeline

    def run(self, messages, **kwargs):
        prompt = "\n".join([msg.content for msg in messages])
        response = self.hf_pipeline(prompt, max_length=50, num_return_sequences=1)
        return {"replies": [response[0]['generated_text']]}
    
# Define the llm for RAG 
llm = HFPipelineGenerator(generator)

#Create a pipeline instance 
rag_pipeline=Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm.messages")
 
# Define a function to handle text generation with translation
def translate_n_generate_text(prompt):
    """
    Translate the input prompt from French to English, generate a response in English, and translate it back to French.
    Args:
        prompt (str): The input text prompt in French.
    Returns:
       str: The generated response translated back to French
    """
    if not prompt:
        return "Please enter a valid prompt."
    else :
        
        # Student asks in French
        question_in_fr= prompt.strip()
        question_in_en= fr_to_en(question_in_fr,
                                max_lenght=100,
                                num_return_sequences=1,
                                repetition_penalty=1.3)[0]['translation_text']
        
        # RAG pipeline to get context-aware answer
        question=question_in_en.strip()
        results= rag_pipeline.run(
            {
                "retriever":{
                    "query": question
                },
                "prompt_builder":{
                    "question":question
                },
            }
        )

        # Get the generated answer in English
        answer_en=results['llm'][0].generated_text
        
        # Translate answer back to French
        answer_fr=en_to_fr(answer_en.strip(),
                        num_return_sequences=1,
                        repetition_penalty=1.3)
        # Return the translated answer

        return answer_fr[0]['translated_text']

        
        
# Create a Gradio interface for the text generation function
root= gr.Interface(
    fn=translate_n_generate_text, 
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs=gr.Textbox(label="Your answer!"),
    title = "StudPal AI",
    description= "An AI-powered study companion that helps you with your studies by generating text based on your prompts."
)
# Launch the Gradio interface
root.launch()

