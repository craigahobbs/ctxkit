# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

import io
import os
import unittest
import unittest.mock

import urllib3

from ctxkit.main import DEFAULT_SYSTEM, main

from .test_main import create_test_files


class TestGemini(unittest.TestCase):

    def test_gemini(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'data: {"candidates": [{"content": {"parts": [{"text": "Goodbye"}]}}]}',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_system(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'data: {"candidates": [{"content": {"parts": [{"text": "Goodbye"}]}}]}',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp'])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ],
                'systemInstruction': {'parts': [{'text': DEFAULT_SYSTEM}]}
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_output(self):
        with create_test_files([
                 ('test.txt', 'test text')
             ]) as temp_dir, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            test_path = os.path.join(temp_dir, 'test.txt')

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'data: {"candidates": [{"content": {"parts": [{"text": "Goodbye"}]}}]}',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            main(['-m', 'Hello', '-i', test_path, '--api', 'gemini', 'gemini-2.0-flash-exp', '-o', test_path, '-s', ''])

            with open(test_path, 'r', encoding='utf-8') as output:
                test_text = output.read()
            self.assertEqual(test_text, 'Goodbye\n')

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello\n\ntest text'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_temperature(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'data: {"candidates": [{"content": {"parts": [{"text": "Goodbye"}]}}]}',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '--temp', '0.2', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ],
                'generationConfig': {
                    'temperature': 0.2
                }
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_top_p(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'data: {"candidates": [{"content": {"parts": [{"text": "Goodbye"}]}}]}',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '--topp', '0.2', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ],
                'generationConfig': {
                    'topP': 0.2
                }
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_max_tokens(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'data: {"candidates": [{"content": {"parts": [{"text": "Goodbye"}]}}]}',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '--maxtok', '100', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ],
                'generationConfig': {
                    'maxOutputTokens': 100
                }
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_empty(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = []

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_no_text(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'data: {"candidates": [{"content": {"parts": [{"text": "Goodbye\\n"}]}}]}',
                b'data: {"candidates": [{"content": {"parts": [{}]}}]}',
                b'data: {"candidates": [{"content": {"parts": [{"text": "Goodbye2"}]}}]}',
                b'data: {"candidates": []}',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\nGoodbye2\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_multiline(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'''\
data: {"candidates": [{"content": {"parts": [{"text": "Goodbye\\n"}]}}]}
data: {"candidates": [{"content": {"parts": [{"text": "Goodbye2"}]}}]}
''',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\nGoodbye2\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_split_chunk(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'''
data: {"candidates": [{"content": {"parts":
data:  [{"text": "Goodbye"}]}}]}
''',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_stdin(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')), \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'data: {"candidates": [{"content": {"parts": [{"text": "Goodbye"}]}}]}',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            main(['--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_stdin_output(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')), \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'data: {"candidates": [{"content": {"parts": [{"text": "Goodbye"}]}}]}',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            with create_test_files([]) as temp_dir:
                output_path = os.path.join(temp_dir, 'output.txt')
                main(['--api', 'gemini', 'gemini-2.0-flash-exp', '-o', output_path, '-s', ''])
                with open(output_path, 'r', encoding='utf-8') as output:
                    output_text = output.read()
            self.assertEqual(output_text, 'Goodbye\n')

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_stdin_error(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')), \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 500
            mock_gemini_response.data = b''

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: Gemini API failed with status 500\n')


    def test_gemini_error(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 500
            mock_gemini_response.data = b''

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: Gemini API failed with status 500\n')


    def test_gemini_error_with_json_message(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 400
            mock_gemini_response.data = \
                b'{"error": {"message": "Invalid request", "status": "INVALID_ARGUMENT", "code": 400}}'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(
            stderr.getvalue(),
            '\nError: Gemini API failed with status 400: Invalid request (status: INVALID_ARGUMENT) (code: 400)\n'
        )


    def test_gemini_error_with_code_only(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 400
            mock_gemini_response.data = b'{"error": {"code": 400}}'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: Gemini API failed with status 400 (code: 400)\n')


    def test_gemini_error_with_string_error(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 500
            mock_gemini_response.data = b'{"error": "Server error occurred"}'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: Gemini API failed with status 500: Server error occurred\n')


    def test_gemini_streaming_error(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gemini_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gemini_response.status = 200
            mock_gemini_response.read_chunked.return_value = [
                b'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}',
                b'data: {"error": {"message": "Rate limit exceeded", "status": "RESOURCE_EXHAUSTED"}}',
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gemini_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=XXXX&alt=sse',
            headers={
                'Content-Type': 'application/json'
            },
            json={
                'contents': [
                    {'role': 'user', 'parts': [{'text': 'Hello'}]}
                ]
            },
            preload_content=False,
            retries=0
        )
        mock_gemini_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Hello')
        self.assertEqual(stderr.getvalue(), '\nError: Gemini API streaming error: Rate limit exceeded (status: RESOURCE_EXHAUSTED)\n')


    def test_gemini_no_api_key(self):
        with unittest.mock.patch('urllib3.PoolManager'), \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gemini', 'gemini-2.0-flash-exp', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: GOOGLE_API_KEY environment variable not set\n')


    def test_gemini_list(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the models API call
            mock_models_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_models_response.status = 200
            mock_models_response.json.return_value = {
                'models': [
                    {'name': 'models/gemini-2.0-flash-exp', 'supportedGenerationMethods': ['generateContent']},
                    {'name': 'models/gemini-1.5-pro', 'supportedGenerationMethods': ['generateContent']},
                    {'name': 'models/gemini-1.5-flash', 'supportedGenerationMethods': ['generateContent']},
                    {'name': 'models/embedding-001', 'supportedGenerationMethods': ['embedContent']}
                ]
            }

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_models_response

            main(['--list', 'gemini'])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='GET',
            url='https://generativelanguage.googleapis.com/v1beta/models?key=XXXX',
            headers={
                'Content-Type': 'application/json'
            },
            retries=0
        )
        mock_models_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '''\
gemini-1.5-flash
gemini-1.5-pro
gemini-2.0-flash-exp
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_gemini_list_error(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the models API call
            mock_models_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_models_response.status = 500
            mock_models_response.data = b''

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_models_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--list', 'gemini'])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='GET',
            url='https://generativelanguage.googleapis.com/v1beta/models?key=XXXX',
            headers={
                'Content-Type': 'application/json'
            },
            retries=0
        )
        mock_models_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: Gemini API failed with status 500\n')


    def test_gemini_list_error_with_json(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('os.environ', {'GOOGLE_API_KEY': 'XXXX'}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the models API call
            mock_models_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_models_response.status = 401
            mock_models_response.data = \
                b'{"error": {"message": "API key not valid", "status": "UNAUTHENTICATED", "code": 401}}'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_models_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--list', 'gemini'])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='GET',
            url='https://generativelanguage.googleapis.com/v1beta/models?key=XXXX',
            headers={
                'Content-Type': 'application/json'
            },
            retries=0
        )
        mock_models_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(
            stderr.getvalue(),
            '\nError: Gemini API failed with status 401: API key not valid (status: UNAUTHENTICATED) (code: 401)\n'
        )


    def test_gemini_list_no_api_key(self):
        with unittest.mock.patch('urllib3.PoolManager'), \
             unittest.mock.patch('os.environ', {}), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--list', 'gemini'])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: GOOGLE_API_KEY environment variable not set\n')
