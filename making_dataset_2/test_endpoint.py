from urllib import request, parse
import json
import os


# curl http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-m21-vllm:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer random-api-key" \
#   -d '{
#         "model": "MiniMax-M2.1",
#         "messages": [
#           {"role": "system", "content": "You are a helpful assistant."},
#           {"role": "user", "content": "Hello! How does a curl request work?"}
#         ]
#       }'


# curl http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-m21-vllm:8000/v1/models


# python -m making_dataset_2.eval.diag_bridge \
#     --model MiniMax-M2.1 \
#     --base-url http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-m21-vllm:8000/v1 \
#     --seed-source inventory \
#     --seed 42 --n-candidates 2 --two-doc

def get_openai_embedding(input_text, model="text-embedding-ada-002"):
    """
    Fetches the embedding for the given input text using OpenAI's API.

    Parameters:
    input_text (str): The text string to be embedded.
    model (str): The model to use for embedding. Default is 'text-embedding-ada-002'.

    Returns:
    list: The embedding vector, or None if an error occurs.
    """
    api_key = os.getenv('OPENAI_API_KEY') # your api key here
    url = 'http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-m21-vllm:8000'

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        'input': input_text,
        'model': model
    }

    data = json.dumps(data).encode('utf-8')

    req = request.Request(url, data=data, headers=headers, method='POST')

    try:
        with request.urlopen(req) as response:
            response_body = response.read()
            embedding = json.loads(response_body)['data'][0]['embedding']
            return embedding
    except Exception as e:
        print(f'An error occurred: {e}')
        return None

# Example usage:
embedding = get_openai_embedding("Example text")
print(embedding)```