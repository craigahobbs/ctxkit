# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

import codecs
import json
import os

import urllib3


# Helper function to get an Ollama API URL
def _get_ollama_url(path):
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    return f'{ollama_host}{path}'


# Decode a streamed, newline-delimited JSON (NDJSON) response into individual JSON objects. The
# Ollama API streams one JSON object per line, but HTTP chunk boundaries do not align with those
# lines - a single chunk may carry multiple objects (common with cloud models) or a partial object
# split across chunks. Buffer the decoded text and yield each complete JSON object as it arrives.
def _iter_ndjson(response):
    decoder = json.JSONDecoder()
    text_decoder = codecs.getincrementaldecoder('utf-8')()
    buffer = ''
    for data in response.read_chunked():
        buffer += text_decoder.decode(data)
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                break
            try:
                chunk, index = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                # Incomplete object - wait for the next chunk to complete it
                break
            buffer = buffer[index:]
            yield chunk

    # The stream ended mid-object - the response was truncated or malformed
    if buffer.strip():
        raise urllib3.exceptions.HTTPError(f'Invalid streamed response: {buffer.strip()!r}')


# Call the Ollama API and yield the response chunk strings
def ollama_chat(pool_manager, model, system_prompt, prompt, temperature=None, top_p=None, max_tokens=None):
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
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})
    data_chat = {
        'model': model,
        'messages': messages,
        'stream': True,
        'think': is_thinking,
    }
    if temperature is not None or top_p is not None or max_tokens is not None:
        data_chat['options'] = {}
    if temperature is not None:
        data_chat['options']['temperature'] = temperature
    if top_p is not None:
        data_chat['options']['top_p'] = top_p
    if max_tokens is not None:
        data_chat['options']['num_predict'] = max_tokens
    response_chat = pool_manager.request('POST', url_chat, json=data_chat, preload_content=False, retries=0)
    try:
        if response_chat.status != 200:
            raise urllib3.exceptions.HTTPError(f'Unknown model "{model}" ({response_chat.status})')

        # Respond with each streamed JSON chunk
        for chunk in _iter_ndjson(response_chat):
            if 'error' in chunk:
                raise urllib3.exceptions.HTTPError(chunk['error'])
            content = chunk['message']['content']
            if content:
                yield content
    finally:
        response_chat.close()


# List available Ollama models
def ollama_list(pool_manager):
    url_tags = _get_ollama_url('/api/tags')
    response_tags = pool_manager.request('GET', url_tags, retries=0)
    try:
        if response_tags.status != 200:
            raise urllib3.exceptions.HTTPError(f'Ollama API failed with status {response_tags.status}')
        data = response_tags.json()
        return [model['name'] for model in data.get('models', [])]
    finally:
        response_tags.close()
