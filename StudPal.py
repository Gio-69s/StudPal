from transformers import pipeline
import gradio as gr 
import torch
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

# Initialize the text generation pipeline with a specific task and model
task = "text2text-generation"
model = "google/flan-t5-base"

# Loading the model with a specified temperature for generation
generator = pipeline(task,
                    model,
                    torch_dtype=torch.float16,  # Use float16 for better performance on compatible hardware
                    device=0,
                    temperature=0.7, # Adjust temperature for creativity and precision in responses
                    trust_remote_code=True, # Trust remote code for model compatibility
                    )
# Define a function to handle text generation
def generate_text(prompt):
    """
    Generate text based on the provided prompt using the initialized pipeline.
    Args:
        prompt (str): The input text prompt for text generation.
    Returns:
       str: The generated text response.
    """
    if not prompt:
        return "Please enter a valid prompt."
    else :
        # Generate text using the pipeline
        response = generator(prompt.strip(),
                            max_length=50,
                            num_return_sequences=1,
                            repetition_penalty=1.3
                            )
    return response[0]['generated_text']

# Create a Gradio interface for the text generation function
root= gr.Interface(
    fn=generate_text, 
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs=gr.Textbox(label="Your answer!"),
    title = "StudPal AI",
    description= "An AI-powered study companion that helps you with your studies by generating text based on your prompts."
)
# Launch the Gradio interface
root.launch(share=True)