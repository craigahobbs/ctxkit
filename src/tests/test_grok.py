# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

import io
import os
import unittest
import unittest.mock

import urllib3

from ctxkit.main import DEFAULT_SYSTEM, main

from .test_main import create_test_files


class TestGrok(unittest.TestCase):

    def test_grok(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": "Goodbye"}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name', '-s', ''])

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

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_system(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": "Goodbye"}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name'])

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
                    {'role': 'system', 'content': DEFAULT_SYSTEM},
                    {'role': 'user', 'content': 'Hello'}
                ],
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_grok_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_output(self):
        with create_test_files([
                 ('test.txt', 'test text')
             ]) as temp_dir, \
             unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            test_path = os.path.join(temp_dir, 'test.txt')

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": "Goodbye"}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '-i', test_path, '--grok', 'model-name', '-o', test_path, '-s', ''])

            with open(test_path, 'r', encoding='utf-8') as output:
                test_text = output.read()
            self.assertEqual(test_text, 'Goodbye\n')

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
                    {'role': 'user', 'content': 'Hello\n\ntest text'}
                ],
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_grok_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_temperature(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": "Goodbye"}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name', '--temp', '0.2', '-s', ''])

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
                'temperature': 0.2,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_grok_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_top_p(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": "Goodbye"}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name', '--topp', '0.2', '-s', ''])

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
                'top_p': 0.2,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_grok_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_max_tokens(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": "Goodbye"}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name', '--maxtok', '100', '-s', ''])

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
                'max_tokens': 100,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_grok_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_empty(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = []

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name', '-s', ''])

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

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_no_content(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": "Goodbye\\n"}}]}',
                b'data: {"choices": [{"delta": {}}]}',
                b'data: {"choices": [{"delta": {"content": "Goodbye2"}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name', '-s', ''])

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

        self.assertEqual(stdout.getvalue(), 'Goodbye\nGoodbye2\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_multiline(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'''\
data: {"choices": [{"delta": {"content": "Goodbye\\n"}}]}
data: {"choices": [{"delta": {"content": "Goodbye2"}}]}
''',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name', '-s', ''])

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

        self.assertEqual(stdout.getvalue(), 'Goodbye\nGoodbye2\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_split_chunk(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'''
data: {"choices": [{"delta":
data:  {"content": "Goodbye"}}]}
''',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['-m', 'Hello', '--grok', 'model-name', '-s', ''])

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

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_stdin(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')) as stdout, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": "Goodbye"}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            main(['--grok', 'model-name', '-s', ''])

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

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_stdin_output(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')) as stdout, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 200
            mock_grok_response.read_chunked.return_value = [
                b'data: {"choices": [{"delta": {"content": "Goodbye"}}]}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            with create_test_files([]) as temp_dir:
                output_path = os.path.join(temp_dir, 'output.txt')
                main(['--grok', 'model-name', '-o', output_path, '-s', ''])
                with open(output_path, 'r', encoding='utf-8') as output:
                    output_text = output.read()
            self.assertEqual(output_text, 'Goodbye\n')

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

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_stdin_error(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')) as stdout, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--grok', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
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

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: xAI API failed with status 500\n')


    def test_grok_error(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_grok_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_grok_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_grok_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--grok', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
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

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: xAI API failed with status 500\n')


    def test_grok_no_api_key(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', None), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--grok', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: XAI_API_KEY environment variable not set\n')


    def test_grok_list(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the models API call
            mock_models_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_models_response.status = 200
            mock_models_response.json.return_value = {
                'data': [
                    {'id': 'grok-beta'},
                    {'id': 'grok-vision-beta'},
                    {'id': 'grok-2-1212'},
                    {'id': 'grok-2-vision-1212'}
                ]
            }

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_models_response

            main(['--list', 'grok'])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='GET',
            url='https://api.x.ai/v1/models',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            retries=0
        )
        mock_models_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '''\
grok-2-1212
grok-2-vision-1212
grok-beta
grok-vision-beta
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_grok_list_error(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the models API call
            mock_models_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_models_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_models_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--list', 'grok'])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='GET',
            url='https://api.x.ai/v1/models',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            retries=0
        )
        mock_models_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: xAI API failed with status 500\n')


    def test_grok_list_no_api_key(self):
        with unittest.mock.patch('ctxkit.grok.XAI_API_KEY', None), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--list', 'grok'])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: XAI_API_KEY environment variable not set\n')
