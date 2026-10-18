import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
import retrieval


# Modèle de prompt réutilisable pour répondre à une question à partir
# d'un contexte fourni, en demandant au modèle de signaler ses incertitudes.
qa_template = '''Tu es un assistant d'études précieux pour un étudiant de BAC ivoirien. 
Utilises le contexte suivant pour répondre à la question de manière claire et précise. 
Si tu ne sais pas, dis-le - ne ments surtout pas.

Context :
{context}

Question : {question}

Réponse : '''

# Associe les emplacements du modèle aux valeurs à fournir lors de son utilisation.
qa_prompt = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question"],
)

formatted = qa_prompt.format(
    context = " ".join(retrieval.chunk_text) , 
    question = input()
)



# Charge les variables d'environnement, notamment la clé d'accès à OpenRouter.
load_dotenv()

# Le client OpenAI est configuré pour communiquer avec l'API compatible
# d'OpenRouter ; la clé est lue depuis OPENROUTER_API_KEY.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1" ,
    api_key=os.environ["OPENROUTER_API_KEY"]

)

# Demande au modèle d'expliquer un concept de mathématiques de Terminale D
# en français accessible et rigoureux.
response = client.chat.completions.create(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    messages=[
        # Le message système définit le rôle et le style pédagogique attendus.
        {
            "role": "system",
            "content": (
                "Tu es StudPal, un camarade pédagogue. "
                "Explique les mathématiques de Terminale D "
                "en français simple, avec rigueur."
            ),
        },

        {
            "role":"user" ,
            "user" : qa_template,
        }

    ],
    # Limite la longueur maximale de la réponse générée.
    max_tokens=500 ,
)

 
