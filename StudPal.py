from transformers import pipeline

# Initialize the text generation pipeline with a specific task and model
task = "text-generation"
model = "openai-community/gpt2"
# Loading the model with a specified temperature for generation
generator = pipeline(task, model , temperature= 0.7)
def generate_text(prompt):
    """
    Generate text based on the provided prompt using the transformers pipeline.
    
    Args:
        prompt (str): The input text to generate from.
        
    Returns:
        str: The generated text.
    """
    # Generate text using the pipeline
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']
