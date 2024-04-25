import os
from utils import read_txt, read_txt_lines
from paths import api_key_path

import voyageai
from openai import OpenAI
from mistralai.client import MistralClient
import cohere
import vertexai

def voyage_api():
    API_KEY = read_txt('%s/voyage_key.txt'%api_key_path)
    API_KEY = API_KEY[:-1]
    os.environ['VOYAGE_API_KEY'] = API_KEY
    client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"),)
    return client
    
def openai_api():
    API_KEY = read_txt('%s/openai_key.txt'%api_key_path)
    API_KEY = API_KEY[:-1]
    os.environ['OPENAI_API_KEY'] = API_KEY
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return client

def mistral_api():
    API_KEY = read_txt('%s/mistral_key.txt'%api_key_path)
    API_KEY = API_KEY[:-1]
    os.environ['MISTRAL_API_KEY'] = API_KEY
    client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))
    return client

def cohere_api():
    API_KEY = read_txt('%s/cohere_key.txt'%api_key_path)
    API_KEY = API_KEY[:-1]    
    os.environ['COHERE_API_KEY'] = API_KEY
    client = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))
    return client

def google_vertex_api():
    KEYS = read_txt_lines('%s/google_vertex.txt'%api_key_path)
    PROJECT_ID = KEYS[0]
    REGION = KEYS[1]
    vertexai.init(project=PROJECT_ID, location=REGION)
