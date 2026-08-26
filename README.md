# StudPal

StudPal is an AI-powered study assistant built to help students learn faster, understand difficult concepts, and work more efficiently with their course materials.

Instead of giving generic answers, StudPal uses Retrieval-Augmented Generation (RAG) to search through relevant course documents, extract the most useful information, and generate context-aware explanations tailored to the student’s question.

## Why this project exists

Students often struggle with three things:

- finding the right information inside long notes and PDFs,
- understanding complicated subjects in a simple way,
- practicing with exercises and feedback without needing constant teacher support.

StudPal was created to solve exactly that by combining AI, document retrieval, and pedagogical assistance in one tool.

---

## Main features

### 1. Smart question answering with RAG

StudPal does not rely only on a general model. It retrieves the most relevant passages from the uploaded document before generating an answer.

This makes the answers more accurate, relevant, and anchored in the student’s actual course content.

Benefits:

- better answers for course-specific questions,
- less hallucination,
- more trust in the generated explanations,
- direct use of notes, chapters, and PDFs as study material.

### 2. PDF ingestion and document processing

The app can load PDF documents, split them into manageable chunks, and convert them into embeddings for semantic search.

This allows the assistant to find the most relevant information even in long academic documents.

### 3. Semantic search with FAISS

Using FAISS as a vector database, StudPal stores embeddings and retrieves the nearest matching information for each question.

This makes the retrieval step fast and efficient, especially when the corpus is large.

### 4. French-first interface

The assistant is currently designed primarily for French-language use, which matches the working environment and user context.

This includes:

- French user prompts,
- French responses and explanations,
- document-driven answers tailored to the local academic context.

### 5. Exercise generation

StudPal is designed to go beyond simple Q&A. It can generate academic exercises based on a course topic and provide detailed corrections.

This is especially useful for:

- revision before exams,
- practice sessions,
- strengthening understanding through applied examples,
- learning by doing rather than only reading.

### 6. Context-aware explanations

The assistant responds with explanations based on the retrieved context, which makes it more pedagogical and useful for learning than a generic chatbot.

It can help with:

- concept explanations,
- summaries,
- comparison between ideas,
- step-by-step reasoning,
- study guidance.

### 7. Extensible architecture

The project is built with a modular structure that can evolve easily:

- PDF loading,
- chunking,
- embeddings,
- vector indexing,
- retrieval,
- LLM answer generation.

This makes it easy to add new features later.

---

## Tech stack

- Python
- LangChain
- Hugging Face Transformers
- SentenceTransformers
- FAISS
- PyPDF
- Hugging Face Models / LLM pipelines
- dotenv for environment configuration

---

## How StudPal works

1. A PDF or course document is loaded.
2. The document is split into smaller text chunks.
3. Each chunk is transformed into embeddings.
4. Similarity search retrieves the most relevant sections.
5. The language model generates a response using both the retrieved context and the user question.
6. The result is a study-focused answer, explanation, or practice exercise.

This pipeline is the foundation of the project’s Retrieval-Augmented Generation approach.

---

## Project structure

- `StudPal.py` — main project logic and pipeline
- `datas/` — course documents and educational materials
- `my_vector_store/` — saved FAISS vector database
- `README.md` — project documentation
- `requirement.txt` — Python dependencies

---

## Getting started

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirement.txt
   ```

4. Place your PDF files in the `datas/` folder.
5. Run the project:

   ```bash
   python StudPal.py
   ```

6. Ask a question or generate a learning exercise using the project’s study workflow.

---

## Example use cases

- “Explain the derivative of a function in a simple way.”
- “Summarize the key points of this chapter.”
- “Generate a practice exercise on probability with a full correction.”
- “Answer this question using only the uploaded course notes.”
- “Translate this concept between French and English.”

---

## Future improvements

The next evolution of StudPal could include:

- conversation memory for multi-turn learning sessions,
- multiple PDF support in a single project,
- a real web interface for easier student use,
- personalized learning recommendations,
- automatic summary generation for notes and chapters,
- stronger academic tutoring features such as quiz generation and progress tracking.

---

## Contact

For questions, feedback, or collaboration, contact:

[giovanni.adadja@gmail.com](mailto:giovanni.adadja@gmail.com)

---

*StudPal is designed as a modern AI study companion for students who want faster understanding, better revision, and smarter learning support.*
