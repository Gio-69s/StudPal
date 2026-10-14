import os
from dotenv import load_dotenv
from openai import OpenAI

# Charge les variables d'environnement définies dans le fichier .env.
load_dotenv()

# Configure le client OpenAI pour utiliser l'API OpenRouter.
# La clé est lue depuis la variable d'environnement OPENROUTER_API_KEY.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1" ,
    api_key=os.environ["OPENROUTER_API_KEY"]

)

# Envoie au modèle les consignes pédagogiques et la question à expliquer.
response = client.chat.completions.create(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    messages=[
        {
            "role": "system",
            "content": (
                "Tu es StudPal, un camarade pédagogue. "
                "Explique les mathématiques de Terminale D "
                "en français simple, avec rigueur."
            ),
        },
        {
            "role": "user",
            "content": "Explique moi le théoreme des valeurs intermediaires."
        },
    ],
    max_tokens=500,
)

# Affiche le texte de la réponse générée par le modèle.
print(response.choices[0].message.content)
