import os
from dotenv import load_dotenv
import voyageai
from openai import OpenAI
from mistralai.client import MistralClient
import cohere
import vertexai

# Load environment variables from .env file
load_dotenv()

def voyage_api():
    client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    return client
    
def openai_api():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

def mistral_api():
    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
    return client

def cohere_api():
    client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
    return client

def google_vertex_api():
    vertexai.init(project=os.getenv("GOOGLE_PROJECT_ID"), location=os.getenv("GOOGLE_REGION"))
