# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

"""
Google Gemini API utilities
"""

import itertools
import json
import os

import urllib3


# Get the Google API key
def get_api_key():
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key is None:
        raise urllib3.exceptions.HTTPError('GOOGLE_API_KEY environment variable not set')
    return api_key


# API endpoint
GEMINI_URL_TEMPLATE = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent'
GEMINI_MODELS_URL = 'https://generativelanguage.googleapis.com/v1beta/models'


# Helper function to format Gemini API errors
def _format_gemini_error(base_message, error_data=None):
    error_message = base_message
    if error_data is not None:
        error_info = error_data.get('error')
        if isinstance(error_info, dict):
            if 'message' in error_info:
                error_message += f': {error_info["message"]}'
            if 'status' in error_info:
                error_message += f' (status: {error_info["status"]})'
            if 'code' in error_info:
                error_message += f' (code: {error_info["code"]})'
        else:
            error_message += f': {error_info}'

    return error_message


# Call the Gemini API and yield the response chunk strings
def gemini_chat(pool_manager, model, system_prompt, prompt, temperature=None, top_p=None, max_tokens=None):
    # Make POST request with streaming
    api_key = get_api_key()
    contents = [{'role': 'user', 'parts': [{'text': prompt}]}]
    gemini_json = {
        'contents': contents
    }
    if system_prompt:
        gemini_json['systemInstruction'] = {'parts': [{'text': system_prompt}]}
    generation_config = {}
    if temperature is not None:
        generation_config['temperature'] = temperature
    if top_p is not None:
        generation_config['topP'] = top_p
    if max_tokens is not None:
        generation_config['maxOutputTokens'] = max_tokens
    if generation_config:
        gemini_json['generationConfig'] = generation_config

    url = GEMINI_URL_TEMPLATE.format(model=model)
    response = pool_manager.request(
        method='POST',
        url=f'{url}?key={api_key}&alt=sse',
        headers={
            'Content-Type': 'application/json'
        },
        json=gemini_json,
        preload_content=False,
        retries=0
    )
    try:
        if response.status != 200:
            error_data = None
            try:
                error_data = json.loads(response.data.decode('utf-8'))
            except:
                pass
            error_message = _format_gemini_error(f'Gemini API failed with status {response.status}', error_data)
            raise urllib3.exceptions.HTTPError(error_message)

        # Process the streaming response
        data_prefix = None
        for line in itertools.chain.from_iterable(line.decode('utf-8').splitlines() for line in response.read_chunked()):
            # Skip non-data lines
            if not line.startswith('data: '):
                continue
            data = line[6:]

            # Combine with previous partial line
            if data_prefix:
                data = data_prefix + data
                data_prefix = None

            # Parse the chunk
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                # If JSON parsing fails, save as prefix for next iteration
                data_prefix = data
                continue

            # Check for errors in the stream
            if 'error' in chunk:
                error_message = _format_gemini_error('Gemini API streaming error', chunk)
                raise urllib3.exceptions.HTTPError(error_message)

            # Yield the chunk content
            candidates = chunk.get('candidates', [])
            if candidates:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                for part in parts:
                    text = part.get('text')
                    if text:
                        yield text

    finally:
        response.close()


# List available Gemini models
def gemini_list(pool_manager):
    api_key = get_api_key()
    response = pool_manager.request(
        method='GET',
        url=f'{GEMINI_MODELS_URL}?key={api_key}',
        headers={
            'Content-Type': 'application/json'
        },
        retries=0
    )
    try:
        if response.status != 200:
            error_data = None
            try:
                error_data = json.loads(response.data.decode('utf-8'))
            except:
                pass
            error_message = _format_gemini_error(f'Gemini API failed with status {response.status}', error_data)
            raise urllib3.exceptions.HTTPError(error_message)
        data = response.json()
        return [
            model['name'].split('/')[-1] for model in data.get('models', [])
            if 'generateContent' in model.get('supportedGenerationMethods', [])
        ]
    finally:
        response.close()
