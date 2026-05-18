# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

"""
Grok API utilities
"""

import json
import os

import urllib3

from ._sse import iter_sse_events


# Get the xAI API key
def get_api_key():
    api_key = os.getenv('XAI_API_KEY')
    if api_key is None:
        raise urllib3.exceptions.HTTPError('XAI_API_KEY environment variable not set')
    return api_key


# API endpoint
XAI_URL = 'https://api.x.ai/v1/chat/completions'
XAI_MODELS_URL = 'https://api.x.ai/v1/models'


# Helper function to format xAI API errors
def _format_xai_error(base_message, error_data=None):
    error_message = base_message
    if error_data is not None:
        error_info = error_data.get('error')
        if isinstance(error_info, dict):
            if 'message' in error_info:
                error_message += f': {error_info["message"]}'
            if 'type' in error_info:
                error_message += f' (type: {error_info["type"]})'
            if 'code' in error_info:
                error_message += f' (code: {error_info["code"]})'
        else:
            error_message += f': {error_info}'

    return error_message


# Call the xAI API and yield the response chunk strings
def grok_chat(pool_manager, model, system_prompt, prompt, temperature=None, top_p=None, max_tokens=None):
    # Make POST request with streaming
    api_key = get_api_key()
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})
    xai_json = {
        'model': model,
        'messages': messages,
        'stream': True
    }
    if temperature is not None:
        xai_json['temperature'] = temperature
    if top_p is not None:
        xai_json['top_p'] = top_p
    if max_tokens is not None:
        xai_json['max_tokens'] = max_tokens
    response = pool_manager.request(
        method='POST',
        url=XAI_URL,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        },
        json=xai_json,
        preload_content=False,
        retries=0
    )
    try:
        if response.status != 200:
            error_data = None
            try:
                error_data = json.loads(response.data.decode('utf-8'))
            except Exception:
                pass
            error_message = _format_xai_error(f'xAI API failed with status {response.status}', error_data)
            raise urllib3.exceptions.HTTPError(error_message)

        # Process the streaming response
        finish_reason = None
        saw_done = False
        for event in iter_sse_events(response):
            if event == '[DONE]':
                saw_done = True
                break

            # Check for errors in the stream
            if 'error' in event:
                error_message = _format_xai_error('xAI API streaming error', event)
                raise urllib3.exceptions.HTTPError(error_message)

            # Track finish_reason for end-of-stream verification (final chunk carries it)
            choice = event['choices'][0]
            if choice.get('finish_reason'):
                finish_reason = choice['finish_reason']

            # Yield the chunk content
            content = choice['delta'].get('content')
            if content:
                yield content

        # Detect silent truncation: server-signalled non-stop, or dropped connection (no [DONE])
        if finish_reason is not None and finish_reason != 'stop':
            raise urllib3.exceptions.HTTPError(f'xAI API response truncated (finish_reason: {finish_reason})')
        if not saw_done:
            raise urllib3.exceptions.HTTPError('xAI API stream ended unexpectedly without [DONE] terminator')

    finally:
        response.close()


# List available Grok models
def grok_list(pool_manager):
    api_key = get_api_key()
    response = pool_manager.request(
        method='GET',
        url=XAI_MODELS_URL,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        retries=0
    )
    try:
        if response.status != 200:
            error_data = None
            try:
                error_data = json.loads(response.data.decode('utf-8'))
            except Exception:
                pass
            error_message = _format_xai_error(f'xAI API failed with status {response.status}', error_data)
            raise urllib3.exceptions.HTTPError(error_message)
        data = response.json()
        return [model['id'] for model in data.get('data', [])]
    finally:
        response.close()
