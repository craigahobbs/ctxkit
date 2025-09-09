# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

"""
Grok API utilities
"""

import itertools
import json
import os

import urllib3


# Load environment variables
XAI_API_KEY = os.getenv('XAI_API_KEY')

# API endpoint
XAI_URL = 'https://api.x.ai/v1/chat/completions'


def grok_chat(pool_manager, model, prompt, temperature):
    # No API key?
    if XAI_API_KEY is None:
        raise urllib3.exceptions.HTTPError('XAI_API_KEY environment variable not set')

    # Make POST request with streaming
    response = pool_manager.request(
        method='POST',
        url=XAI_URL,
        headers={
            'Authorization': f'Bearer {XAI_API_KEY}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        },
        json={
            'model': model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': temperature,
            'stream': True
        },
        preload_content=False,
        retries=0
    )
    try:
        if response.status != 200:
            raise urllib3.exceptions.HTTPError(f'xAI API failed with status {response.status}')

        # Process the streaming response
        data_prefix = None
        for line in itertools.chain.from_iterable(line.decode('utf-8').splitlines() for line in response.read_chunked()):
            # Parse the data chunk
            if not line.startswith('data: '):
                continue
            data = line[6:]
            if data == '[DONE]':
                break

            # Combine with previous partial line
            if data_prefix:
                data = data_prefix + data
                data_prefix = None

            # Parse the chunk
            try:
                chunk = json.loads(data)
            except:
                data_prefix = data
                continue

            # Yield the chunk content
            content = chunk['choices'][0]['delta'].get('content')
            if content:
                yield content

    finally:
        response.close()
