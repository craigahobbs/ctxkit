# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import json
import os

import urllib3


# Helper function to get an Ollama API URL
def _get_ollama_url(path):
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    return f'{ollama_host}{path}'


# Send a prompt string to the Ollama API and yield the response chunk strings
def ollama_chat(pool_manager, model, prompt, temperature=None):
    # Is this a thinking model?
    url_show = _get_ollama_url('/api/show')
    data_show = {'model': model}
    response_show = pool_manager.request('POST', url_show, json=data_show, retries=0)
    try:
        if response_show.status != 200:
            raise urllib3.exceptions.HTTPError(f'Unknown model "{model}" ({response_show.status})')
        model_show = response_show.json()
    finally:
        response_show.close()
    is_thinking = 'capabilities' in model_show and 'thinking' in model_show['capabilities']

    # Start a streaming chat request
    url_chat = _get_ollama_url('/api/chat')
    data_chat = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'options': {
            'temperature': temperature
        },
        'stream': True,
        'think': is_thinking,
    }
    response_chat = pool_manager.request('POST', url_chat, json=data_chat, preload_content=False, retries=0)
    try:
        if response_chat.status != 200:
            raise urllib3.exceptions.HTTPError(f'Unknown model "{model}" ({response_chat.status})')

        # Respond with each streamed JSON chunk
        for chunk in (json.loads(line.decode('utf-8')) for line in response_chat.read_chunked()):
            if 'error' in chunk:
                raise urllib3.exceptions.HTTPError(chunk['error'])
            content = chunk['message']['content']
            if content:
                yield content
    finally:
        response_chat.close()
