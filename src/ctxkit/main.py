# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

"""
ctxkit command-line script main module
"""

import argparse
import os
import shutil
import sys

import urllib3

from .api import API_PROVIDERS, DEFAULT_SYSTEM, DEFAULT_SYSTEM_DIFF, output_api_call
from .config import CTXKIT_SMD, fetch_text, process_config, process_config_items


def main(argv=None):
    """
    ctxkit command-line script main entry point
    """

    # Combine the command-line and environment arguments
    argv_env = os.getenv('CTXKIT_FLAGS', '').split()
    argv_combined = argv_env + (sys.argv[1:] if argv is None else argv)

    # Compute the API provider documentation
    api_doc_lines = []
    api_desc_indent = max(len(api) for api in API_PROVIDERS) + 1
    for api in sorted(API_PROVIDERS.keys()):
        api_doc_lines.append(f'  {api}{" " * (api_desc_indent - len(api))}- {API_PROVIDERS[api]["description"]}')
    api_doc = '\n'.join(api_doc_lines)

    # Command line arguments
    argument_parser_args = {'prog': 'ctxkit'}
    if sys.version_info >= (3, 14): # pragma: no cover
        argument_parser_args['color'] = False
    parser = argparse.ArgumentParser(**argument_parser_args)
    parser.add_argument('-g', '--config-help', action='store_true', help='display the JSON configuration file format')
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('-e', '--extract', action='store_true', help='extract response files')
    output_group.add_argument('--diff', action='store_true', help='use unified diff format for file changes')
    output_group.add_argument('-o', '--output', metavar='PATH', help='output to the file path')
    output_group.add_argument('-b', '--backup', action='store_true', help='backup output files with ".bak" extension')
    items_group = parser.add_argument_group('Prompt Items')
    items_group.add_argument('-c', '--config', metavar='PATH', dest='items', action=TypedItemAction, item_type='config',
                             help='process the JSON configuration file path or URL')
    items_group.add_argument('-m', '--message', metavar='TEXT', dest='items', action=TypedItemAction, item_type='message',
                             help='add a prompt message')
    items_group.add_argument('-i', '--include', metavar='PATH', dest='items', action=TypedItemAction, item_type='include',
                             help='add the file path or URL text')
    items_group.add_argument('-t', '--template', metavar='PATH', dest='items', action=TypedItemAction, item_type='template',
                             help='add the file path or URL template text')
    items_group.add_argument('-f', '--file', metavar='PATH', dest='items', action=TypedItemAction, item_type='file',
                             help='add the file path or URL as a text file')
    items_group.add_argument('-d', '--dir', metavar='PATH', dest='items', action=TypedItemAction, item_type='dir',
                             help="add a directory's text files")
    items_group.add_argument('-v', '--var', nargs=2, metavar=('VAR', 'EXPR'), dest='items', action=TypedItemAction, item_type='var',
                             help='define a variable (reference with "{{var}}")')
    items_group.add_argument('-s', '--system', metavar='PATH', help='the system prompt file path or URL, "" for none')
    dir_group = parser.add_argument_group('Directory Options')
    dir_group.add_argument('-x', '--ext', action='append', default=[], help='add a directory text file extension')
    dir_group.add_argument('-l', '--depth', metavar='INT', type=int, default=0, help='the maximum directory depth, default is 0 (infinite)')
    api_group = parser.add_argument_group('API Calling')
    api_group.add_argument('--api', nargs=2, metavar=('API', 'MODEL'), action=APIAction,
                           help='pass to an API provider (see "API Providers")')
    api_group.add_argument('--list', metavar='API', action=APIAction,
                           help='list API provider models (see "API Providers")')
    api_group.add_argument('--temp', metavar='NUM', type=float, help='set the model response temperature')
    api_group.add_argument('--topp', metavar='NUM', type=float, help='set the model response top_p')
    api_group.add_argument('--maxtok', metavar='NUM', type=int, help='set the model response max tokens')
    api_group.add_argument('--noapi', dest='api', action='store_false', help='do not pass to an API provider')
    parser.epilog = f'''\
API Providers:
{api_doc}

Examples:
  ctxkit --api ollama qwen3.6:35b -m "How do I count code lines?"
  ctxkit --api grok grok-4.3 -f README.md -f main.py -f test_main.py -m "Add a -q argument" -e
  ctxkit --api claude claude-opus-4-7 -f README.md -d src -x py -i spec.txt -e
  ctxkit --list grok
'''
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    args = parser.parse_args(args=argv_combined)

    # Show configuration file format?
    if args.config_help:
        print(CTXKIT_SMD.strip())
        return

    # Initialize urllib3 PoolManager
    pool_manager = urllib3.PoolManager()

    try:
        # List models?
        if args.list:
            models = API_PROVIDERS[args.list]['list'](pool_manager)
            print('\n'.join(sorted(models)))
            return

        # Load the config file
        config = {'items': []}
        for item_type, item_value in (args.items or []):
            if item_type == 'config':
                config['items'].append({'config': item_value})
            elif item_type == 'include':
                config['items'].append({'include': item_value})
            elif item_type == 'template':
                config['items'].append({'template': item_value})
            elif item_type == 'file':
                config['items'].append({'file': item_value})
            elif item_type == 'dir':
                config['items'].append({'dir': {'path': item_value, 'exts': args.ext, 'depth': args.depth}})
            elif item_type == 'var':
                config['items'].append({'var': {'name': item_value[0], 'value': item_value[1]}})
            else: # item_type == 'message':
                config['items'].append({'message': item_value})

        # Get the system prompt
        if args.system is not None:
            system_prompt = fetch_text(pool_manager, args.system) if args.system else None
        elif args.diff:
            system_prompt = DEFAULT_SYSTEM_DIFF
        else:
            system_prompt = DEFAULT_SYSTEM

        # Output file?
        if args.output:
            # Backup the output file, if requested
            if args.backup and os.path.isfile(args.output):
                shutil.copy(args.output, f'{args.output}.bak')

            # Create the output directory
            output_dir = os.path.dirname(args.output)
            if output_dir: # pragma: no branch
                os.makedirs(output_dir, exist_ok=True)

        # Pass stdin to an AI?
        if args.api and not config['items']:
            prompt = sys.stdin.read()
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as output:
                    output_api_call(args, pool_manager, output, system_prompt, prompt)
            else:
                output_api_call(args, pool_manager, sys.stdout, system_prompt, prompt)
            return

        # No items specified
        if not config['items']:
            parser.error('no prompt items specified')

        # Process the configuration
        if args.api:
            # Pass prompt to an AI
            prompt = process_config(pool_manager, args, config, {})
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as output:
                    output_api_call(args, pool_manager, output, system_prompt, prompt)
            else:
                output_api_call(args, pool_manager, sys.stdout, system_prompt, prompt)
        else:
            # Output to file?
            if args.output:
                prompt = process_config(pool_manager, args, config, {})
                with open(args.output, 'w', encoding='utf-8') as output:
                    print(prompt, file=output)
            else:
                # Output to stdout
                items = []
                if system_prompt:
                    items.append(f'<system>\n{system_prompt}\n</system>')
                items.extend(process_config_items(pool_manager, args, config, {}))
                for ix_item, item_text in enumerate(items):
                    if ix_item != 0:
                        print()
                    print(item_text)

    except Exception as exc:
        print(f'\nError: {exc}', file=sys.stderr)
        sys.exit(2)


# argparse action to validate API provider
class APIAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        provider = values[0] if isinstance(values, list) else values
        if provider not in API_PROVIDERS:
            parser.error(f'Invalid API provider "{provider}". Valid options are: {", ".join(sorted(API_PROVIDERS.keys()))}')
        setattr(namespace, self.dest, values)


# argparse action typed-value items
class TypedItemAction(argparse.Action):

    def __init__(self, *args, **kwargs):
        self.item_type = kwargs.pop('item_type')
        super().__init__(*args, **kwargs)


    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest)
        if items is None:
            items = []
            setattr(namespace, self.dest, items)
        items.append((self.item_type, values))
