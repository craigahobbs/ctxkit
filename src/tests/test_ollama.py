# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

import io
import json
import os
import unittest
import unittest.mock

import urllib3

from ctxkit.main import DEFAULT_SYSTEM, main

from .test_main import create_test_files


class TestOllama(unittest.TestCase):

    def test_ollama(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            main(['-m', 'Hello', '--ollama', 'model-name', '-s', ''])

        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), 'Hi there!\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_ollama_system(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            main(['-m', 'Hello', '--ollama', 'model-name'])

        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'system', 'content': DEFAULT_SYSTEM},
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), 'Hi there!\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_ollama_output(self):
        with create_test_files([
                 ('test.txt', 'test text')
             ]) as temp_dir, \
             unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            test_path = os.path.join(temp_dir, 'test.txt')

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            main(['-m', 'Hello', '-i', test_path, '--ollama', 'model-name', '-o', test_path, '-s', ''])

            with open(test_path, 'r', encoding='utf-8') as output:
                test_text = output.read()
            self.assertEqual(test_text, 'Hi there!\n')

        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello\n\ntest text'}
                        ],
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_ollama_temperature(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            main(['-m', 'Hello', '--ollama', 'model-name', '--temp', '0.2', '-s', ''])

        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'options': {
                            'temperature': 0.2
                        },
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), 'Hi there!\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_ollama_top_p(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            main(['-m', 'Hello', '--ollama', 'model-name', '--topp', '0.2', '-s', ''])

        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'options': {
                            'top_p': 0.2
                        },
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), 'Hi there!\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_ollama_max_tokens(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            main(['-m', 'Hello', '--ollama', 'model-name', '--maxtok', '100', '-s', ''])

        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'options': {
                            'num_predict': 100
                        },
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), 'Hi there!\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_ollama_thinking(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': ['thinking']}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': '', 'thinking': 'Hmmm'}}).encode('utf-8'),
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            main(['-m', 'Hello', '--ollama', 'model-name', '-s', ''])

        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'stream': True,
                        'think': True
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), 'Hi there!\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_ollama_stdin(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')) as stdout, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            main(['--ollama', 'model-name', '-s', ''])

        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), 'Hi there!\n')
        self.assertEqual(stderr.getvalue(), '')


    def test_ollama_stdin_output(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')) as stdout, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'message': {'content': 'there!'}}).encode('utf-8')
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            with create_test_files([]) as temp_dir:
                output_path = os.path.join(temp_dir, 'output.txt')
                main(['--ollama', 'model-name', '-o', output_path, '-s', ''])
                with open(output_path, 'r', encoding='utf-8') as output:
                    output_text = output.read()
            self.assertEqual(output_text, 'Hi there!\n')

        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_ollama_stdin_empty(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')) as stdout, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = []

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            main(['--ollama', 'model-name', '-s', ''])

        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '')


    def test_ollama_stdin_error_chunk(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdin', io.StringIO('Hello')) as stdout, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'error': 'BOOM!'}).encode('utf-8')
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            with self.assertRaises(SystemExit) as cm_exc:
                main(['--ollama', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), 'Hi ')
        self.assertEqual(stderr.getvalue(), '\nError: BOOM!\n')


    def test_ollama_show_error(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.return_value = mock_show_response

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--ollama', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(mock_pool_manager_instance.request.call_count, 1)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_not_called()
        mock_show_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: Unknown model "model-name" (500)\n')


    def test_ollama_chat_error(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 500

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--ollama', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_not_called()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(stderr.getvalue(), '\nError: Unknown model "model-name" (500)\n')


    def test_ollama_chunk_error(self):
        with unittest.mock.patch('urllib3.PoolManager') as mock_pool_manager, \
             unittest.mock.patch('sys.stdout', io.StringIO()) as stdout, \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:

            # Create a mock Response object for the show API call
            mock_show_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_show_response.status = 200
            mock_show_response.json.return_value = {'capabilities': []}

            # Create a mock Response object for the chat API call
            mock_chat_response = unittest.mock.Mock(spec=urllib3.response.HTTPResponse)
            mock_chat_response.status = 200
            mock_chat_response.read_chunked.return_value = [
                json.dumps({'message': {'content': 'Hi '}}).encode('utf-8'),
                json.dumps({'error': 'BOOM!'}).encode('utf-8')
            ]

            # Configure the mock PoolManager instance
            mock_pool_manager_instance = mock_pool_manager.return_value
            mock_pool_manager_instance.request.side_effect = [mock_show_response, mock_chat_response]

            with self.assertRaises(SystemExit) as cm_exc:
                main(['-m', 'Hello', '--ollama', 'model-name', '-s', ''])

        self.assertEqual(cm_exc.exception.code, 2)
        self.assertEqual(mock_pool_manager_instance.request.call_count, 2)
        self.assertListEqual(
            mock_pool_manager_instance.request.call_args_list,
            [
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/show',
                    json={'model': 'model-name'},
                    retries=0
                ),
                unittest.mock.call(
                    'POST',
                    'http://127.0.0.1:11434/api/chat',
                    json={
                        'model': 'model-name',
                        'messages': [
                            {'role': 'user', 'content': 'Hello'}
                        ],
                        'stream': True,
                        'think': False
                    },
                    preload_content=False,
                    retries=0
                )
            ]
        )
        mock_show_response.json.assert_called_once_with()
        mock_chat_response.read_chunked.assert_called_once_with()
        mock_show_response.close.assert_called_once_with()
        mock_chat_response.close.assert_called_once_with()

        self.assertEqual(stdout.getvalue(), 'Hi ')
        self.assertEqual(stderr.getvalue(), '\nError: BOOM!\n')
