# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="openai-community/gpt2")
# Generate text using the pipeline
response= pipe("Hello, how are you?", max_length=50, num_return_sequences=1)
print(response)