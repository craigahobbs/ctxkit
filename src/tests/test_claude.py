# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

import io
import os
import unittest
import unittest.mock

import urllib3

from ctxkit.main import DEFAULT_SYSTEM, main

from .test_main import create_test_files


class TestClaude(unittest.TestCase):

    def test_claude(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = [
                b'data: {"type": "content_block_delta", "delta": {"text": "Goodbye"}}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            main(['-m', 'Hello', '--api', 'claude', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_system(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = [
                b'data: {"type": "content_block_delta", "delta": {"text": "Goodbye"}}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            main(['-m', 'Hello', '--api', 'claude', 'model-name'])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'max_tokens': 8000,
                'stream': True,
                'system': DEFAULT_SYSTEM
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_output(self):
        with create_test_files([
                 ('test.txt', 'test text')
             ]) as temp_dir, \
             unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            test_path = os.path.join(temp_dir, 'test.txt')

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = [
                b'data: {"type": "content_block_delta", "delta": {"text": "Goodbye"}}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            main(['-m', 'Hello', '-i', test_path, '--api', 'claude', 'model-name', '-o', test_path, '-s', ''])

            with open(test_path, 'r', encoding='utf-8') as output:
                test_text = output.read()
            self.assertEqual(test_text, 'Goodbye\n')

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello\n\ntest text'}
                ],
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_temperature(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = [
                b'data: {"type": "content_block_delta", "delta": {"text": "Goodbye"}}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            main(['-m', 'Hello', '--api', 'claude', 'model-name', '--temp', '0.2', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'temperature': 0.2,
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_top_p(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = [
                b'data: {"type": "content_block_delta", "delta": {"text": "Goodbye"}}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            main(['-m', 'Hello', '--api', 'claude', 'model-name', '--topp', '0.2', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'top_p': 0.2,
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_empty(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = []

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            main(['-m', 'Hello', '--api', 'claude', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_multiline(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = [
                b'''\
data: {"type": "content_block_delta", "delta": {"text": "Goodbye\\n"}}
data: {"type": "content_block_delta", "delta": {"text": "Goodbye2"}}
''',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            main(['-m', 'Hello', '--api', 'claude', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\nGoodbye2\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_split_chunk(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = [
                b'''
data: {"type": "content_block_delta", "delta":
data: {"text": "Goodbye"}}
''',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            main(['-m', 'Hello', '--api', 'claude', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_stdin(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')), \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = [
                b'data: {"type": "content_block_delta", "delta": {"text": "Goodbye"}}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            main(['--api', 'claude', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_stdin_output(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')), \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = [
                b'data: {"type": "content_block_delta", "delta": {"text": "Goodbye"}}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            with create_test_files([]) as temp_dir:
                output_path = os.path.join(temp_dir, 'output.txt')
                main(['--api', 'claude', 'model-name', '-o', output_path, '-s', ''])
                with open(output_path, 'r', encoding='utf-8') as output:
                    output_text = output.read()
            self.assertEqual(output_text, 'Goodbye\n')

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_stdin_error(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')), \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--api', 'claude', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: Claude API failed with status 500\n')


    def test_claude_error(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'claude', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: Claude API failed with status 500\n')


    def test_claude_no_api_key(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', None), \
             unittest.mock.patch('urllib3.PoolManager'), \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'claude', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: ANTHROPIC_API_KEY environment variable not set\n')


    def test_claude_false_cases(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = [
                b'data: {"type": "other_type", "data": "ignored"}',  # False case: wrong type
                b'data: {"type": "content_block_delta", "delta": {"text": "Valid"}}',  # True case
                b'data: {"type": "content_block_delta", "delta": {}}',  # False case: no 'text' in delta
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            main(['-m', 'Hello', '--api', 'claude', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        # Only the valid text should be output
        self.assertEqual(stdout.getvalue(), 'Valid\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_api_error_event(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_claude_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_claude_response.status = 200
            mock_claude_response.read_chunked.return_value = [
                b'data: {"type": "error", "error": {"message": "Test API error"}}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_claude_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'claude', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'messages': [
                    {'role': 'user', 'content': 'Hello'}
                ],
                'max_tokens': 8000,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_claude_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: Claude API error: Test API error\n')


    def test_claude_list(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the models API call
            mock_models_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_models_response.status = 200
            mock_models_response.json.return_value = {
                'data': [
                    {'id': 'claude-3-opus-20240229'},
                    {'id': 'claude-3-sonnet-20240229'},
                    {'id': 'claude-3-haiku-20240307'},
                    {'id': 'claude-2.1'}
                ]
            }

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_models_response

            main(['--list', 'claude'])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='GET',
            url='https://api.anthropic.com/v1/models',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            retries=0
        )
        mock_models_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '''\
claude-2.1
claude-3-haiku-20240307
claude-3-opus-20240229
claude-3-sonnet-20240229
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_claude_list_error(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the models API call
            mock_models_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_models_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_models_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--list', 'claude'])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='GET',
            url='https://api.anthropic.com/v1/models',
            headers={
                'x-api-key': 'XXXX',
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            retries=0
        )
        mock_models_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: Claude API failed with status 500\n')


    def test_claude_list_no_api_key(self):
        with unittest.mock.patch('ctxkit.claude.ANTHROPIC_API_KEY', None), \
             unittest.mock.patch('urllib3.PoolManager'), \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--list', 'claude'])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: ANTHROPIC_API_KEY environment variable not set\n')
