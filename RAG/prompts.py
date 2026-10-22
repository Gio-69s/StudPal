# Module standard utilisé pour lire la clé d'API depuis les variables d'environnement.
import os
# Charge les variables définies dans un fichier .env.
from dotenv import load_dotenv
# Fournit le client pour envoyer des requêtes à une API compatible avec OpenAI.
from openai import OpenAI
# Construit un modèle de prompt à partir d'un texte et de variables à remplir.
from langchain_core.prompts import PromptTemplate
# Récupère les extraits de contexte associés à une question.
from retrieval import retrieving_and_context_chunk


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

# Récupère la question saisie par l'utilisateur dans le terminal.
Query = input()
# Recherche le contexte pertinent, puis réunit les extraits en une seule chaîne.
Background = " ".join(retrieving_and_context_chunk(Query))

# Remplace les variables du modèle de prompt par la question et le contexte.
formatted = qa_prompt.format(
     context= Background , 
     question = Query
)



# Charge les variables d'environnement, notamment la clé d'accès à OpenRouter.
load_dotenv()

# Initialise le client HTTP avec l'URL de l'API OpenRouter et la clé configurée.
# Le client OpenAI est configuré pour communiquer avec l'API compatible
# d'OpenRouter ; la clé est lue depuis OPENROUTER_API_KEY.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1" ,
    api_key=os.environ["OPENROUTER_API_KEY"]

)

# Envoie le prompt au modèle et récupère sa réponse.
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

        # Transmet au modèle le prompt complété avec la question et le contexte.
        {
            "role":"user" ,
            "content" : formatted,
        }

    ],
    # Limite la longueur maximale de la réponse générée.
    max_tokens=500 ,
)

# Affiche le texte de la réponse générée par le modèle.
print(response.choices[0].message.content)


 
