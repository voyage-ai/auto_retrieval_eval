from setup_api import openai_api
client = openai_api()

def get_gpt4_results(input_text, generative_model):
    response = client.chat.completions.create(
            model=generative_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
        )
    
    response_text = response.choices[0].message.content
    return response_text
