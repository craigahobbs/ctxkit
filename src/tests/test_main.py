# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

from contextlib import contextmanager
import io
import json
import os
from tempfile import TemporaryDirectory
import unittest
import unittest.mock

import urllib3

import ctxkit.__main__
from ctxkit.main import DEFAULT_SYSTEM, main


# Helper context manager to create a list of files in a temporary directory
@contextmanager
def create_test_files(file_defs):
    tempdir = TemporaryDirectory()
    try:
        for path_parts, content in file_defs:
            if isinstance(path_parts, str):
                path_parts = [path_parts]
            path = os.path.join(tempdir.name, *path_parts)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as file_:
                file_.write(content)
        yield tempdir.name
    finally:
        tempdir.cleanup()


class TestMain(unittest.TestCase):

    def test_main_submodule(self):
        self.assertTrue(ctxkit.__main__)


    def test_help_config(self):
        with unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            main(['-g'])
        self.assertTrue(stdout.getvalue().startswith('# The ctxkit configuration file format'))
        self.assertEqual(stderr.getvalue(), '')


    def test_system_default(self):
        with unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            main(['-m', 'Hello', '-m', 'Goodbye'])
        self.assertEqual(stdout.getvalue(), f'''\
<system>
{DEFAULT_SYSTEM}
</system>

Hello

Goodbye
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_system_path(self):
        with create_test_files([
                 ('system.txt', 'You are a friendly assistant')
             ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            system_path = os.path.join(temp_dir, 'system.txt')
            main(['-m', 'Hello', '-m', 'Goodbye', '-s', system_path])
        self.assertEqual(stdout.getvalue(), '''\
<system>
You are a friendly assistant
</system>

Hello

Goodbye
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_no_items(self):
        with unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as cm_exc:
                main([])
        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertTrue(stderr.getvalue().endswith('ctxkit: error: no prompt items specified\n'))


    def test_output(self):
        with create_test_files([]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            output_path = os.path.join(temp_dir, 'output.txt')
            backup_path = f'{output_path}.bak'

            main(['-m', 'Hello', '-m', 'Goodbye', '-o', output_path, '-s', ''])

            with open(output_path, 'r', encoding='utf-8') as output:
                output_text = output.read()

            self.assertFalse(os.path.isfile(backup_path))

        self.assertEqual(output_text, '''\
Hello

Goodbye
''')
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_output_backup(self):
        with create_test_files([
            ('output.txt', 'Hello')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            output_path = os.path.join(temp_dir, 'output.txt')
            backup_path = f'{output_path}.bak'

            main(['-m', 'Hello', '-m', 'Goodbye', '-o', output_path, '-b', '-s', ''])

            with open(output_path, 'r', encoding='utf-8') as output:
                output_text = output.read()

            with open(backup_path, 'r', encoding='utf-8') as backup:
                backup_text = backup.read()

        self.assertEqual(output_text, '''\
Hello

Goodbye
''')
        self.assertEqual(backup_text, 'Hello')
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_output_update(self):
        with create_test_files([
                ('test.txt', 'test text')
             ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            test_path = os.path.join(temp_dir, 'test.txt')
            main(['-m', 'Hello', '-i', test_path, '-o', test_path, '-s', ''])
            with open(test_path, 'r', encoding='utf-8') as output:
                output_text = output.read()
        self.assertEqual(output_text, '''\
Hello

test text
''')
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_extract(self):
        with create_test_files([
                 ('file.txt', 'File #0')
             ]) as temp_dir, \
             unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'file.txt')
            file_path2 = os.path.join(temp_dir, 'file2.txt')
            response = f'''\
<{file_path}>
File #1
</{file_path}>

<{file_path2}>
File #2
</{file_path2}>
'''

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": ' + json.dumps(response).encode('utf-8') + b'}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name', '--extract', '-s', ''])

            with open(file_path, 'r', encoding='utf-8') as file_:
                file_text = file_.read()
            self.assertEqual(file_text, 'File #1')

            self.assertFalse(os.path.exists(f'{file_path}.bak'))

            with open(file_path2, 'r', encoding='utf-8') as file_:
                file_text2 = file_.read()
            self.assertEqual(file_text2, 'File #2')

            self.assertFalse(os.path.exists(f'{file_path2}.bak'))

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.x.ai/v1/chat/completions',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_grok_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), f'{response}\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_extract_backup(self):
        with create_test_files([
                 ('file.txt', 'File #0')
             ]) as temp_dir, \
             unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'file.txt')
            file_path2 = os.path.join(temp_dir, 'file2.txt')
            response = f'''\
<{file_path}>
File #1
</{file_path}>

<{file_path2}>
File #2
</{file_path2}>
'''

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": ' + json.dumps(response).encode('utf-8') + b'}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name', '--extract', '-b', '-s', ''])

            with open(file_path, 'r', encoding='utf-8') as file_:
                file_text = file_.read()
            self.assertEqual(file_text, 'File #1')

            with open(f'{file_path}.bak', 'r', encoding='utf-8') as file_:
                file_text = file_.read()
            self.assertEqual(file_text, 'File #0')

            with open(file_path2, 'r', encoding='utf-8') as file_:
                file_text2 = file_.read()
            self.assertEqual(file_text2, 'File #2')

            self.assertFalse(os.path.exists(f'{file_path2}.bak'))

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.x.ai/v1/chat/completions',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_grok_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), f'{response}\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_extract_delete(self):
        with create_test_files([
                 ('file.txt', 'Test')
             ]) as temp_dir, \
             unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'file.txt')
            file_path2 = os.path.join(temp_dir, 'file2.txt')
            response = f'''\
<{file_path}>
ctxkit: delete
</{file_path}>

<{file_path2}>
ctxkit: delete
</{file_path2}>
'''

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": ' + json.dumps(response).encode('utf-8') + b'}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name', '--extract', '-b', '-s', ''])

            self.assertFalse(os.path.exists(file_path))
            self.assertFalse(os.path.exists(f'{file_path}.bak'))
            self.assertFalse(os.path.exists(file_path2))
            self.assertFalse(os.path.exists(f'{file_path2}.bak'))

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.x.ai/v1/chat/completions',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_grok_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), f'{response}\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_message(self):
        with unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            main(['-m', 'Hello', '-m', 'Goodbye', '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello

Goodbye
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_config(self):
        with create_test_files([
            ('test.json', json.dumps({
                'items': [
                    {'long': ['Hello,', 'message!']},
                    {'long': ['Hello,', 'long!']}
                ]
            }))
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            main(['-c', os.path.join(temp_dir, 'test.json'), '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello,
message!

Hello,
long!
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_config_first_outer(self):
        with create_test_files([
            ('test.json', json.dumps({
                'items': [
                    {'message': 'test1'},
                    {'config': 'test2.json'}
                ]
            })),
            ('test2.json', json.dumps({
                'items': [
                    {'long': ['test2']}
                ]
            }))
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            main(['-c', os.path.join(temp_dir, 'test.json'), '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
test1

test2
''')
        self.assertEqual(stderr.getvalue(), '')



    def test_config_first_inner(self):
        with create_test_files([
            ('test.json', json.dumps({
                'items': [
                    {'config': 'test2.json'},
                    {'message': 'test1'}
                ]
            })),
            ('test2.json', json.dumps({
                'items': [
                    {'long': ['test2']}
                ]
            }))
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            main(['-c', os.path.join(temp_dir, 'test.json'), '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
test2

test1
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_config_failure(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            # Create a mock Response object for the pull request
            mock_pull_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_pull_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_pull_response

            unknown_path = os.path.join('not-found', 'unknown.json')
            with self.assertRaises(SystemExit) as cm_exc:
                main(['-c', unknown_path, '-c', 'https://test.local/unknown.json', '-s', ''])
        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), f'\nError: [Errno 2] No such file or directory: {unknown_path!r}\n')


    def test_config_url(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            # Create a mock Response object for the pull request
            mock_pull_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_pull_response.status = 200
            mock_pull_response.data = b'{"items": [{"message": "Hello!"}]}'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_pull_response

            main(['-c', 'https://invalid.local/test.json', '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello!
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_config_nested(self):
        with create_test_files([
            ('main.json', json.dumps({
                'items': [
                    {'config': os.path.join('subdir', 'nested.json')},
                    {'message': 'Main message'}
                ]
            })),
            (('subdir', 'nested.json'), json.dumps({
                'items': [
                    {'file': 'nested.txt'}
                ]
            })),
            (('subdir', 'nested.txt'), 'Nested message')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            nested_path = os.path.join(temp_dir, 'subdir', 'nested.txt')
            main(['-c', os.path.join(temp_dir, 'main.json'), '-s', ''])
        self.assertEqual(stdout.getvalue(), f'''\
<{nested_path}>
Nested message
</{nested_path}>

Main message
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_config_invalid(self):
        with create_test_files([
            ('invalid.json', '{"items": [{"invalid": "value"}]}')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as cm_exc:
                main(['-c', os.path.join(temp_dir, 'invalid.json'), '-s', ''])
        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), "\nError: Unknown member 'items.0.invalid'\n")


    def test_include(self):
        with create_test_files([
            ('test.txt', 'Hello!'),
            ('test2.txt', 'Hello2!')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            file_path2 = os.path.join(temp_dir, 'test2.txt')
            main(['-i', file_path, '-i', file_path2, '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello!

Hello2!
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_include_url(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            # Create a mock Response object for the pull request
            mock_pull_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_pull_response.status = 200
            mock_pull_response.data = b'URL content\n'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_pull_response

            main(['-i', 'https://test.local', '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
URL content
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_include_variable(self):
        with create_test_files([
            ('test.txt', 'Hello!')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path_var = os.path.join(temp_dir, '{{name}}.txt')
            main(['-v', 'name', 'test', '-i', file_path_var, '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello!
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_include_empty(self):
        with create_test_files([
            ('test.txt', '')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            main(['-i', file_path, '-s', ''])
        self.assertEqual(stdout.getvalue(), '\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_include_strip(self):
        with create_test_files([
            ('test.txt', '\nHello!\n')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            main(['-i', file_path, '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello!
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_include_error(self):
        with unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            unknown_path = os.path.join('not-found', 'unknown.txt')
            with self.assertRaises(SystemExit) as cm_exc:
                main(['-i', unknown_path, '-s', ''])
        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), f"\nError: [Errno 2] No such file or directory: {unknown_path!r}\n")


    def test_template(self):
        with create_test_files([
            ('test.txt', 'Hello, {{name}}!')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            main(['-v', 'name', 'World', '-t', file_path, '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello, World!
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_template_missing(self):
        with create_test_files([
            ('test.txt', 'Hello, {{name}}!')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            main(['-t', file_path, '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello, !
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_template_url(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            # Create a mock Response object for the pull request
            mock_pull_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_pull_response.status = 200
            mock_pull_response.data = b'Hello, {{name}}!\n'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_pull_response

            main(['-v', 'name', 'World', '-t', 'https://test.local', '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello, World!
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_template_empty(self):
        with create_test_files([
            ('test.txt', '')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            main(['-t', file_path, '-s', ''])
        self.assertEqual(stdout.getvalue(), '\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_template_strip(self):
        with create_test_files([
            ('test.txt', '\nHello!\n')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            main(['-t', file_path, '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello!
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_template_error(self):
        with unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            unknown_path = os.path.join('not-found', 'unknown.txt')
            with self.assertRaises(SystemExit) as cm_exc:
                main(['-t', unknown_path, '-s', ''])
        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), f"\nError: [Errno 2] No such file or directory: {unknown_path!r}\n")


    def test_file(self):
        with create_test_files([
            ('test.txt', 'Hello!'),
            ('test2.txt', 'Hello2!')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            file_path2 = os.path.join(temp_dir, 'test2.txt')
            main(['-f', file_path, '-f', file_path2, '-s', ''])
        self.assertEqual(stdout.getvalue(), f'''\
<{file_path}>
Hello!
</{file_path}>

<{file_path2}>
Hello2!
</{file_path2}>
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_file_url(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            # Create a mock Response object for the pull request
            mock_pull_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_pull_response.status = 200
            mock_pull_response.data = b'URL content\n'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_pull_response

            main(['-f', 'https://test.local', '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
<https://test.local>
URL content
</https://test.local>
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_file_url_error(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            # Create a mock Response object for the pull request
            mock_pull_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_pull_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_pull_response

            with self.assertRaises(SystemExit):
                main(['-f', 'https://test.local', '-s', ''])
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: POST https://test.local failed with status 500\n')


    def test_file_variable(self):
        with create_test_files([
            ('test.txt', 'Hello!')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            file_path_var = os.path.join(temp_dir, '{{name}}.txt')
            main(['-v', 'name', 'test', '-f', file_path_var, '-s', ''])
        self.assertEqual(stdout.getvalue(), f'''\
<{file_path}>
Hello!
</{file_path}>
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_file_empty(self):
        with create_test_files([
            ('test.txt', '')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            main(['-f', file_path, '-s', ''])
        self.assertEqual(stdout.getvalue(), f'''\
<{file_path}>
</{file_path}>
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_file_strip(self):
        with create_test_files([
            ('test.txt', '\nHello!\n')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            main(['-f', file_path, '-s', ''])
        self.assertEqual(stdout.getvalue(), f'''\
<{file_path}>
Hello!
</{file_path}>
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_file_error(self):
        with unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            unknown_path = os.path.join('not-found', 'unknown.txt')
            with self.assertRaises(SystemExit) as cm_exc:
                main(['-f', unknown_path, '-s', ''])
        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), f"\nError: [Errno 2] No such file or directory: {unknown_path!r}\n")


    def test_dir(self):
        with create_test_files([
            ('test.txt', 'Hello!'),
            (('subdir', 'sub.txt'), 'Goodbye!')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            sub_path = os.path.join(temp_dir, 'subdir', 'sub.txt')
            main(['-d', temp_dir, '-x', 'txt', '-s', ''])
        self.assertEqual(stdout.getvalue(), f'''\
<{file_path}>
Hello!
</{file_path}>

<{sub_path}>
Goodbye!
</{sub_path}>
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_dir_variable(self):
        with create_test_files([
            (('subdir', 'sub.txt'), 'Goodbye!'),
            (('subdir2', 'sub2.txt'), 'Goodbye2!')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            sub_path = os.path.join(temp_dir, 'subdir', 'sub.txt')
            sub_dir_var = os.path.join(temp_dir, '{{name}}')
            main(['-v', 'name', 'subdir', '-d', sub_dir_var, '-x', 'txt', '-s', ''])
        self.assertEqual(stdout.getvalue(), f'''\
<{sub_path}>
Goodbye!
</{sub_path}>
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_dir_depth(self):
        with create_test_files([
            ('test.txt', 'Hello!'),
            (('subdir', 'sub.txt'), 'Goodbye!')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            main(['-d', temp_dir, '-x', 'txt', '-l', '1', '-s', ''])
        self.assertEqual(stdout.getvalue(), f'''\
<{file_path}>
Hello!
</{file_path}>
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_dir_empty(self):
        with create_test_files([
            ('test.txt', '')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            file_path = os.path.join(temp_dir, 'test.txt')
            main(['-d', temp_dir, '-x', 'txt', '-s', ''])
        self.assertEqual(stdout.getvalue(), f'''\
<{file_path}>
</{file_path}>
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_dir_no_files(self):
        with create_test_files([
            (('subdir', 'file1.md'), 'Content1')
        ]) as temp_dir, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as cm_exc:
                main(['-d', temp_dir, '-x', 'txt', '-s', ''])
        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), f'\nError: No files found, "{temp_dir}"\n')


    def test_dir_relative_not_found(self):
        with unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            unknown_path = os.path.join('not-found', 'unknown')
            unknown_path2 = os.path.join('not-found', 'unknown2')
            with self.assertRaises(SystemExit) as cm_exc:
                main(['-d', unknown_path, '-d', unknown_path2, '-x', 'txt', '-s', ''])
        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertTrue(stderr.getvalue().endswith(f'{unknown_path!r}\n'))


    def test_variable(self):
        with unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            main(['-v', 'first', 'Foo', '-v', 'Last', 'Bar', '-m', 'Hello, {{first}} {{ Last }}!', '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello, Foo Bar!
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_variable_unknown(self):
        with unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            main(['-v', 'Last', 'Bar', '-m', 'Hello, {{first}} {{ Last }}!', '-s', ''])
        self.assertEqual(stdout.getvalue(), '''\
Hello,  Bar!
''')
        self.assertEqual(stderr.getvalue(), '')
