from transformers import pipeline, AutoTokenizer
import gradio as gr 

# Initialize the text generation pipeline with a specific task and model
task = "text-generation"
model = "openai-community/gpt2"

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model)

# Loading the model with a specified temperature for generation
generator = pipeline(task, model , temperature= 0.7, top_k=50, top_p=0.95)

# Define a function to handle text generation
def generate_text(prompt):
    """
    Generate text based on the provided prompt.
    
    Args:
        prompt (str): The input text to generate a response for.
        
    Returns:
        str: The generated text response.
    """
    # Generate text using the pipeline
    response = generator(prompt.strip(),
                          max_length=50,
                            num_return_sequences=1,
                            repetition_penalty=1.3,
                            eos_token_id=tokenizer.encode("")
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
root.launch()
