import pdb
from setup_api import openai_api
client = openai_api()

def get_gpt4_results(input_text, generative_model):
    try:
        response = client.chat.completions.create(
            model=generative_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
        )
    except:
        pdb.set_trace()
        response = client.chat.completions.create(
            model=generative_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
        )

    response_text = response.choices[0].message.content
    print(response_text)

    # pdb.set_trace()
    return response_text
