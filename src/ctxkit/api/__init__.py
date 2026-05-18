# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

"""
ctxkit API utilities
"""

import os
import re
import shutil

from .claude import claude_chat, claude_list
from ..config import is_url
from ..diff import apply_diff
from .gemini import gemini_chat, gemini_list
from .gpt import gpt_chat, gpt_list
from .grok import grok_chat, grok_list
from .ollama import ollama_chat, ollama_list


# API providers
API_PROVIDERS = {
    'claude': {
        'description': 'Claude (Anthropic) API',
        'chat': claude_chat,
        'list': claude_list
        },
    'gemini': {
        'description': 'Gemini (Google) API',
        'chat': gemini_chat,
        'list': gemini_list
    },
    'gpt': {
        'description': 'ChatGPT (OpenAI) API',
        'chat': gpt_chat,
        'list': gpt_list
    },
    'grok': {
        'description': 'Grok (xAI) API',
        'chat': grok_chat,
        'list': grok_list
    },
    'ollama': {
        'description': 'Ollama API',
        'chat': ollama_chat,
        'list': ollama_list
    }
}


DEFAULT_SYSTEM_PREFIX = '''\
You are a helpful assistant that can read and modify files provided in the prompt.
'''


DEFAULT_SYSTEM_SUFFIX = '''\
To delete a file, use:

<filename>
ctxkit: delete
</filename>

Files containing the inline instructions, lines or code comments that begin with "ctxkit:" followed
by instructions, are modified per the instructions and the inline instructions removed. Only process
instructions specifically prefixed with "ctxkit:" and ignore instructions intended for others (e.g.
"user:").

<filename>
    # ctxkit: Add a docstring
    def foo():
        pass
</filename>

Do not output files that have not changed.
You can include explanatory text outside of these file tags.
'''


DEFAULT_SYSTEM = f'''\
{DEFAULT_SYSTEM_PREFIX}

You can read and modify files provided in the prompt. When outputting modified or new files, always
provide the complete, updated content of the entire file, not just the modified parts. Use this
format:

<filename>
<complete content of the file>
</filename>

{DEFAULT_SYSTEM_SUFFIX}
'''


DEFAULT_SYSTEM_DIFF = f'''\
{DEFAULT_SYSTEM_PREFIX}

Files in the prompt have each line preceded by its line number and a colon (e.g. "1:first line").
Line numbers are for reference only and are not part of the file content.

When outputting modified or new files, provide the changes as a unified diff. Use this format:

<filename>
--- a/filename
+++ b/filename
@@ -start,count +start,count @@
 context line
-removed line
+added line
 context line
</filename>

For new files, use /dev/null as the old file:

<filename>
--- /dev/null
+++ b/filename
@@ -0,0 +1,N @@
+first line
+second line
</filename>

{DEFAULT_SYSTEM_SUFFIX}
'''


# Helper to output the response from stdin to passed to an API
def output_api_call(args, pool_manager, output, system_prompt, prompt):
    provider, model = args.api
    api_func = API_PROVIDERS[provider]['chat']

    # Write the response to the output
    chunks = []
    for chunk in api_func(pool_manager, model, system_prompt, prompt, args.temp, args.topp, args.maxtok):
        chunks.append(chunk)
        output.write(chunk)
        output.flush()
    if chunks:
        output.write('\n')

    # Extract files, if requested
    if args.extract:
        _extract_files(args, ''.join(chunks))


# Helper to extract files from a response
def _extract_files(args, response):
    search_pos = 0
    while True:
        match = _R_FILENAME_TAG.search(response, search_pos)
        if not match:
            break
        file_path = os.path.normpath(match.group(1))
        content = match.group(2).strip()
        search_pos = match.end()

        # Ignore URLs
        if is_url(file_path):
            continue

        # Delete?
        if content == 'ctxkit: delete':
            if os.path.exists(file_path):
                os.remove(file_path)
            continue

        # Backup the existing file
        if args.backup and os.path.exists(file_path):
            shutil.copy(file_path, f'{file_path}.bak')

        # Create the file's parent directory
        file_dir = os.path.dirname(file_path)
        if file_dir: # pragma: no branch
            os.makedirs(file_dir, exist_ok=True)

        # Is content a unified diff?
        if args.diff:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as fh:
                    old_content = fh.read()
            else:
                old_content = ''
            content = apply_diff(old_content, content)

        # Write the file
        with open(file_path, 'w', encoding='utf-8') as file_:
            file_.write(content.strip())
            file_.write('\n')


_R_FILENAME_TAG = re.compile(r'^<([^<>]+)>\n(.*?)\n</\1>', re.DOTALL | re.MULTILINE)
