# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

"""
Shared SSE event parser for streaming API responses.
"""

import codecs
import itertools
import json


# Yield SSE events from a urllib3 chunked response
def iter_sse_events(response):
    """
    Each yielded value is either the literal string '[DONE]' (the sentinel some
    providers send) or a parsed JSON value (typically a dict).

    Handles HTTP chunks that split an event's 'data:' line across boundaries:
    when a 'data:' line fails to parse as JSON, the partial JSON is held until
    the next line arrives and concatenated, whether or not that next line
    carries its own 'data:' prefix.

    Uses an incremental UTF-8 decoder so that multi-byte codepoints split
    across HTTP chunks are reconstructed instead of raising UnicodeDecodeError.
    """
    decoder = codecs.getincrementaldecoder('utf-8')()
    data_prefix = None
    for line in itertools.chain.from_iterable(
        decoder.decode(chunk).splitlines() for chunk in response.read_chunked()
    ):
        if data_prefix is not None and not line.startswith('data: '):
            # Continuation of a partial 'data:' line arriving without its own
            # 'data:' prefix (i.e. the actual chunk-boundary case)
            combined = data_prefix + line
            try:
                yield json.loads(combined)
                data_prefix = None
            except json.JSONDecodeError:
                data_prefix = combined
        elif line.startswith('data: '):
            data = line[6:]
            if data == '[DONE]':
                yield '[DONE]'
            else:
                # Continuation where the next chunk also begins with 'data:'
                if data_prefix is not None:
                    data = data_prefix + data
                    data_prefix = None
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    data_prefix = data
