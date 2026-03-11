from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, TransformersGenerator
from haystack.pipelines import GenerativeQAPipeline

# read file text
path = r"C:\Users\giova\OneDrive\Documents\GitHub\StudPal\datas\Maths\Dossier_Af\resume_essentiel_studpal.txt"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

docs = [Document(content=text, meta={"source": "resume_essentiel_studpal"})]

# use InMemoryDocumentStore (no native FAISS required)
doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs)

retriever = DensePassageRetriever(
    document_store=doc_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=False
)

# update embeddings in the store
doc_store.update_embeddings(retriever)

generator = TransformersGenerator(model_name_or_path="google/flan-t5-base", task="text2text-generation", max_length=200, device=-1)
pipe = GenerativeQAPipeline(retriever=retriever, generator=generator)
res = pipe.run(
    query="J'ai un contrôle sur les primitives, donne-moi l'essentiel et 1 exo",
    params={"Retriever": {"top_k": 4}, "Generator": {"max_length": 180}}
)
print(res["answers"][0].answer)

# Additional code block for retriever testing
try:
    r = DensePassageRetriever(document_store=None, query_embedding_model="facebook/dpr-question_encoder-single-nq-base", passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base", use_gpu=False)
    print("Retriever created")
    print("Query embedding test:", r.embed_queries(["hello"]))
except Exception as e:
    import traceback; traceback.print_exc()
