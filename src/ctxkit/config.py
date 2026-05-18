# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

"""
The ctxkit config file definition and utilities
"""

from functools import partial
import json
import os
import re

import schema_markdown
import urllib3


# Process a configuration model and return the prompt string
def process_config(pool_manager, args, config, variables, root_dir='.'):
    return '\n\n'.join(process_config_items(pool_manager, args, config, variables, root_dir))


# Process a configuration model and yield the prompt item strings
def process_config_items(pool_manager, args, config, variables, root_dir='.'):
    # Output the prompt items
    for item in config['items']:
        item_key = list(item.keys())[0]

        # Get the item path, if any
        item_path = None
        if item_key in ('config', 'include', 'template', 'file'):
            item_path = _replace_variables(item[item_key], variables)
        elif item_key == 'dir':
            item_path = _replace_variables(item[item_key]['path'], variables)

        # Normalize the item path
        if item_path is not None and not is_url(item_path) and not os.path.isabs(item_path):
            item_path = os.path.normpath(os.path.join(root_dir, item_path))

        # Config item
        if item_key == 'config':
            included_config = schema_markdown.validate_type(CTXKIT_TYPES, 'CtxKitConfig', json.loads(fetch_text(pool_manager, item_path)))
            yield from process_config_items(pool_manager, args, included_config, variables, os.path.dirname(item_path))

        # File include item
        elif item_key == 'include':
            yield fetch_text(pool_manager, item_path)

        # File include with variables item
        elif item_key == 'template':
            yield _replace_variables(fetch_text(pool_manager, item_path), variables)

        # File item
        elif item_key == 'file':
            file_text = fetch_text(pool_manager, item_path)
            if args.diff:
                file_text = _add_line_numbers(file_text)
            newline = '\n'
            yield f'<{item_path}>{newline}{file_text}{newline if file_text else ""}</{item_path}>'

        # Directory item
        elif item_key == 'dir':
            # Recursively find the files of the requested extensions
            dir_exts = [f'.{ext.lstrip(".")}' for ext in item['dir'].get('exts') or []]
            dir_depth = item['dir'].get('depth', 0)
            dir_files = list(_get_directory_files(item_path, dir_exts, dir_depth))
            if not dir_files:
                raise Exception(f'No files found, "{item_path}"')

            # Output the file text
            newline = '\n'
            for file_path in dir_files:
                file_text = fetch_text(pool_manager, file_path)
                if args.diff:
                    file_text = _add_line_numbers(file_text)
                yield f'<{file_path}>{newline}{file_text}{newline if file_text else ""}</{file_path}>'

        # Variable definition item
        elif item_key == 'var':
            variables[item['var']['name']] = item['var']['value']

        # Long message item
        elif item_key == 'long':
            yield _replace_variables('\n'.join(item['long']), variables)

        # Message item
        else: # if item_key == 'message'
            yield _replace_variables(item['message'], variables)


# Helper to fetch a file or URL text
def fetch_text(pool_manager, path):
    if is_url(path):
        response = pool_manager.request(method='GET', url=path, retries=0)
        try:
            if response.status != 200:
                raise urllib3.exceptions.HTTPError(f'POST {path} failed with status {response.status}')
            return response.data.decode('utf-8').strip()
        finally:
            response.close()
    else:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read().strip()


# Helper to determine if a path is a URL
def is_url(path):
    return re.match(_R_URL, path)

_R_URL = re.compile(r'^[a-z]+:')


# Helper to replace variable references
def _replace_variables(text, variables):
    return _R_VARIABLE.sub(partial(_replace_variables_match, variables), text)

def _replace_variables_match(variables, match):
    var_name = match.group(1)
    return str(variables.get(var_name, ''))

_R_VARIABLE = re.compile(r'\{\{\s*([_a-zA-Z]\w*)\s*\}\}')


# Helper to add line numbers to file text
def _add_line_numbers(text):
    return '\n'.join(f'{ix + 1}:{line}' for ix, line in enumerate(text.splitlines()))


# Helper enumerator to recursively get a directory's files
def _get_directory_files(dir_name, file_exts, max_depth=0, current_depth=0):
    yield from (file_path for _, file_path in sorted(_get_directory_files_helper(dir_name, file_exts, max_depth, current_depth)))

def _get_directory_files_helper(dir_name, file_exts, max_depth, current_depth):
    # Recursion too deep?
    if max_depth > 0 and current_depth >= max_depth:
        return

    # Scan the directory for files
    with os.scandir(dir_name) as entries:
        for entry in entries:
            if entry.is_file():
                if os.path.splitext(entry.name)[1] in file_exts:
                    file_path = os.path.normpath(os.path.join(dir_name, entry.name))
                    yield (os.path.split(file_path), file_path)
            elif entry.is_dir(): # pragma: no branch
                dir_path = os.path.join(dir_name, entry.name)
                yield from _get_directory_files_helper(dir_path, file_exts, max_depth, current_depth + 1)


# The ctxkit configuration file format
CTXKIT_SMD = '''\
# The ctxkit configuration file format
struct CtxKitConfig

    # The list of prompt items
    CtxKitItem[len > 0] items


# A prompt item
union CtxKitItem

    # Config file path or URL
    string config

    # A prompt message
    string message

    # A long prompt message
    string[len > 0] long

    # File path or URL text
    string include

    # File path or URL template text
    string template

    # File path or URL as a text file
    string file

    # Add a directory's text files
    CtxKitDir dir

    # Set a variable (reference with "{{var}}")
    CtxKitVariable var


# A directory item
struct CtxKitDir

    # The directory file path or URL
    string path

    # The file extensions to include (e.g. ".py")
    string[] exts

    # The directory traversal depth (default is 0, infinite)
    optional int(>= 0) depth


# A variable definition item
struct CtxKitVariable

    # The variable's name
    string name

    # The variable's value
    string value
'''
CTXKIT_TYPES = schema_markdown.parse_schema_markdown(CTXKIT_SMD)
