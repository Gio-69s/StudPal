from transformers import pipeline

# Initialize the text generation pipeline with a specific task and model
task = "text-generation"
model = "openai-community/gpt2"
# Loading the model with a specified temperature for generation
generator = pipeline(task, model , temperature= 0.7)

# Create a loop to continuously accept user input
while True:
    prompt = input("You :")
    # Generate text using the pipeline
    response = generator(prompt.strip(), max_length=50, num_return_sequences=1)
    print(f"StudPal : {response[0]['generated_text']}")
    # Check if the user wanna exit the loop
    if prompt.lower() == 'exit':
        print(" StudPal : See you soon champion!")
        break
        
