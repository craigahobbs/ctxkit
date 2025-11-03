# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

import io
import os
import unittest
import unittest.mock

import urllib3

from ctxkit.main import DEFAULT_SYSTEM, main

from .test_main import create_test_files


class TestGPT(unittest.TestCase):

    def test_gpt(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'data: {"type": "response.output_text.delta", "delta": "Goodbye"}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            main(['-m', 'Hello', '--api', 'gpt', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_system(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'data: {"type": "response.output_text.delta", "delta": "Goodbye"}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            main(['-m', 'Hello', '--api', 'gpt', 'model-name'])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'instructions': DEFAULT_SYSTEM,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_output(self):
        with create_test_files([
                 ('test.txt', 'test text')
             ]) as temp_dir, \
             unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            test_path = os.path.join(temp_dir, 'test.txt')

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'data: {"type": "response.output_text.delta", "delta": "Goodbye"}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            main(['-m', 'Hello', '-i', test_path, '--api', 'gpt', 'model-name', '-o', test_path, '-s', ''])

            with open(test_path, 'r', encoding='utf-8') as output:
                test_text = output.read()
            self.assertEqual(test_text, 'Goodbye\n')

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello\n\ntest text',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_temperature(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'data: {"type": "response.output_text.delta", "delta": "Goodbye"}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            main(['-m', 'Hello', '--api', 'gpt', 'model-name', '--temp', '0.2', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'temperature': 0.2,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_top_p(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'data: {"type": "response.output_text.delta", "delta": "Goodbye"}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            main(['-m', 'Hello', '--api', 'gpt', 'model-name', '--topp', '0.2', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'top_p': 0.2,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_max_tokens(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'data: {"type": "response.output_text.delta", "delta": "Goodbye"}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            main(['-m', 'Hello', '--api', 'gpt', 'model-name', '--maxtok', '100', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'max_output_tokens': 100,
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_empty(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = []

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            main(['-m', 'Hello', '--api', 'gpt', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_no_delta(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'data: {"type": "response.output_text.delta", "delta": "Goodbye\\n"}',
                b'data: {"type": "response.output_text.delta"}',
                b'data: {"type": "response.output_text.delta", "delta": "Goodbye2"}',
                b'data: {"type": "other_event"}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            main(['-m', 'Hello', '--api', 'gpt', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\nGoodbye2\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_multiline(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'''\
data: {"type": "response.output_text.delta", "delta": "Goodbye\\n"}
data: {"type": "response.output_text.delta", "delta": "Goodbye2"}
''',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            main(['-m', 'Hello', '--api', 'gpt', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\nGoodbye2\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_split_chunk(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'''
data: {"type": "response.output_text.delta",
data:  "delta": "Goodbye"}
''',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            main(['-m', 'Hello', '--api', 'gpt', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_stdin(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')) as stdout, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'data: {"type": "response.output_text.delta", "delta": "Goodbye"}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            main(['--api', 'gpt', 'model-name', '-s', ''])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Goodbye\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_stdin_output(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')) as stdout, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'data: {"type": "response.output_text.delta", "delta": "Goodbye"}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            with create_test_files([]) as temp_dir:
                output_path = os.path.join(temp_dir, 'output.txt')
                main(['--api', 'gpt', 'model-name', '-o', output_path, '-s', ''])
                with open(output_path, 'r', encoding='utf-8') as output:
                    output_text = output.read()
            self.assertEqual(output_text, 'Goodbye\n')

        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_stdin_error(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')) as stdout, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 500
            mock_gpt_response.data = b''

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--api', 'gpt', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: OpenAI API failed with status 500\n')


    def test_gpt_error(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 500
            mock_gpt_response.data = b''

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gpt', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: OpenAI API failed with status 500\n')


    def test_gpt_error_with_json_message(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 400
            mock_gpt_response.data = \
                b'{"error": {"message": "Invalid request", "type": "invalid_request_error", "code": "invalid_api_key"}}'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gpt', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(
            stderr.getvalue(),
            '\nError: OpenAI API failed with status 400: Invalid request (type: invalid_request_error) (code: invalid_api_key)\n'
        )


    def test_gpt_error_with_code_only(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 400
            mock_gpt_response.data = b'{"error": {"code": "invalid_api_key"}}'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gpt', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: OpenAI API failed with status 400 (code: invalid_api_key)\n')


    def test_gpt_error_with_string_error(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 500
            mock_gpt_response.data = b'{"error": "Server error occurred"}'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gpt', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: OpenAI API failed with status 500: Server error occurred\n')


    def test_gpt_streaming_error(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the HTTP response
            mock_gpt_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_gpt_response.status = 200
            mock_gpt_response.read_chunked.return_value = [
                b'data: {"type": "response.output_text.delta", "delta": "Hello"}',
                b'data: {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}',
                b'data: [DONE]'
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_gpt_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gpt', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='POST',
            url='https://api.openai.com/v1/responses',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'model-name',
                'input': 'Hello',
                'stream': True
            },
            preload_content=False,
            retries=0
        )
        mock_gpt_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), 'Hello')
        self.assertEqual(stderr.getvalue(), '\nError: OpenAI API streaming error: Rate limit exceeded (type: rate_limit_error)\n')


    def test_gpt_no_api_key(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', None), \
             unittest.mock.patch('urllib3.PoolManager'), \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--api', 'gpt', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: OPENAI_API_KEY environment variable not set\n')


    def test_gpt_list(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the models API call
            mock_models_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_models_response.status = 200
            mock_models_response.json.return_value = {
                'data': [
                    {'id': 'gpt-4-turbo-preview'},
                    {'id': 'gpt-4'},
                    {'id': 'gpt-3.5-turbo'},
                    {'id': 'gpt-4-32k'}
                ]
            }

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_models_response

            main(['--list', 'gpt'])

        mock_pool_manager_instance.request.assert_called_once_with(
            method='GET',
            url='https://api.openai.com/v1/models',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            retries=0
        )
        mock_models_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '''\
gpt-3.5-turbo
gpt-4
gpt-4-32k
gpt-4-turbo-preview
''')
        self.assertEqual(stderr.getvalue(), '')


    def test_gpt_list_error(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
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
                main(['--list', 'gpt'])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='GET',
            url='https://api.openai.com/v1/models',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            retries=0
        )
        mock_models_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: OpenAI API failed with status 500\n')


    def test_gpt_list_error_with_json(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', 'XXXX'), \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the models API call
            mock_models_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_models_response.status = 401
            mock_models_response.data = \
                b'{"error": {"message": "Incorrect API key", "type": "invalid_request_error", "code": "invalid_api_key"}}'

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_models_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--list', 'gpt'])

        self.assertEqual(cm_exc.exception.code, 2)
        mock_pool_manager_instance.request.assert_called_once_with(
            method='GET',
            url='https://api.openai.com/v1/models',
            headers={
                'Authorization': 'Bearer XXXX',
                'Content-Type': 'application/json'
            },
            retries=0
        )
        mock_models_response.close.assert_called_once()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(
            stderr.getvalue(),
            '\nError: OpenAI API failed with status 401: Incorrect API key (type: invalid_request_error) (code: invalid_api_key)\n'
        )


    def test_gpt_list_no_api_key(self):
        with unittest.mock.patch('ctxkit.gpt.OPENAI_API_KEY', None), \
             unittest.mock.patch('urllib3.PoolManager'), \
             unittest.mock.patch.dict('os.environ', {}, clear=True), \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--list', 'gpt'])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: OPENAI_API_KEY environment variable not set\n')
