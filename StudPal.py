from transformers import pipeline
import gradio as gr
import torch


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
    else :
        # Student asks in French
        question_in_fr= prompt.strip()
        question_in_en= fr_to_en(question_in_fr,
                                max_lenght=100,
                                num_return_sequences=1,
                                repetition_penalty=1.3)[0]['translation_text']
        # Generate answer in English
        answer_en=generator(question_in_en.strip(),
                            max_length=100,
                            num_return_sequences=1,
                            repetition_penalty=1.3)[0]['generated_text']
        
        # Translate answer back to French
        answer_fr=en_to_fr(answer_en.strip(),
                        num_return_sequences=1,
                        repetition_penalty=1.3)
        
    return answer_fr[0]['translated_text']


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

