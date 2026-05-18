# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

"""
Claude API utilities
"""

import os

import urllib3

from ._sse import iter_sse_events


# Get the Anthropic API key
def get_api_key():
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key is None:
        raise urllib3.exceptions.HTTPError('ANTHROPIC_API_KEY environment variable not set')
    return api_key


# API endpoint
ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages'
ANTHROPIC_MODELS_URL = 'https://api.anthropic.com/v1/models'


# Anthropic API requires max_tokens
ANTHROPIC_MAX_TOKENS = 8000


# Call the Claude API and yield the response chunk strings
def claude_chat(pool_manager, model, system_prompt, prompt, temperature=None, top_p=None, max_tokens=None):
    # Make POST request with streaming
    api_key = get_api_key()
    messages = [{'role': 'user', 'content': prompt}]
    claude_json = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens or ANTHROPIC_MAX_TOKENS,
        'stream': True
    }
    if system_prompt:
        claude_json['system'] = system_prompt
    if temperature is not None:
        claude_json['temperature'] = temperature
    if top_p is not None:
        claude_json['top_p'] = top_p
    response = pool_manager.request(
        method='POST',
        url=ANTHROPIC_URL,
        headers={
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        },
        json=claude_json,
        preload_content=False,
        retries=0
    )
    try:
        if response.status != 200:
            raise urllib3.exceptions.HTTPError(f'Claude API failed with status {response.status}')

        # Process the streaming response
        stop_reason = None
        saw_terminator = False
        for event in iter_sse_events(response):
            if event == '[DONE]':
                saw_terminator = True
                break

            # Check for API errors in the event
            event_type = event.get('type')
            if event_type == 'error':
                error_message = event.get('error', {}).get('message', 'Unknown API error')
                raise urllib3.exceptions.HTTPError(f'Claude API error: {error_message}')

            # Track stop_reason from message_delta event (carries final stop_reason)
            if event_type == 'message_delta':
                new_stop_reason = event.get('delta', {}).get('stop_reason')
                if new_stop_reason:
                    stop_reason = new_stop_reason

            # message_stop event is the real Anthropic terminator
            if event_type == 'message_stop':
                saw_terminator = True
                break

            # Yield content from content_block_delta
            if event_type == 'content_block_delta' and 'delta' in event:
                delta = event['delta']
                if 'text' in delta:
                    yield delta['text']

        # Detect silent truncation: bad stop_reason, or dropped connection without
        # any completion signal. A known-good stop_reason is itself a clean terminator.
        if stop_reason is not None and stop_reason not in ('end_turn', 'stop_sequence'):
            raise urllib3.exceptions.HTTPError(f'Claude API response truncated (stop_reason: {stop_reason})')
        if stop_reason is None and not saw_terminator:
            raise urllib3.exceptions.HTTPError('Claude API stream ended unexpectedly without terminator')

    finally:
        response.close()


# List available Claude models
def claude_list(pool_manager):
    api_key = get_api_key()
    response = pool_manager.request(
        method='GET',
        url=ANTHROPIC_MODELS_URL,
        headers={
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        },
        retries=0
    )
    try:
        if response.status != 200:
            raise urllib3.exceptions.HTTPError(f'Claude API failed with status {response.status}')
        data = response.json()
        return [model['id'] for model in data.get('data', [])]
    finally:
        response.close()
