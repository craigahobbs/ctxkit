# ctxkit

[![PyPI - Status](https://img.shields.io/pypi/status/ctxkit)](https://pypi.org/project/ctxkit/)
[![PyPI](https://img.shields.io/pypi/v/ctxkit)](https://pypi.org/project/ctxkit/)
[![GitHub](https://img.shields.io/github/license/craigahobbs/ctxkit)](https://github.com/craigahobbs/ctxkit/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ctxkit)](https://pypi.org/project/ctxkit/)

ctxkit is a command-line tool for constructing AI prompts containing files, directories, and URL
content. For example:

```sh
ctxkit -m "Please review the following source code file." -f main.py
```

In the preceding example, the `-m` argument outputs the message text, and the `-f` argument outputs
the `main.py` file text as follows.

```
<system>
...
</system>

Please review the following source code file.

<main.py>
print('Hello, world!')
</main.py>
```


## Installation

To install ctxkit, enter the following commands in a terminal window:

**macOS and Linux**

```
python3 -m venv $HOME/venv --upgrade-deps
. $HOME/venv/bin/activate
pip install ctxkit
```

**Windows**

```
python3 -m venv %USERPROFILE%\venv --upgrade-deps
%USERPROFILE%\venv\Scripts\activate
pip install ctxkit
```


## API Calling

ctxkit supports calling the
[Ollama API](https://ollama.com/)
and the
[Grok (xAI) API](https://docs.x.ai/docs/tutorial)
via the `--ollama` and `--grok` arguments, respectively.

**Ollama**

```sh
ctxkit -m 'Hello!' --ollama gpt-oss:20b
```

**Grok**

Click here to create an [xAI API key](https://docs.x.ai/docs/tutorial).


```sh
export XAI_API_KEY=<key>
ctxkit -m 'Hello!' --grok grok-3
```

You can call an API with a prompt from `stdin` by passing no prompt items:

```sh
echo 'Hello!' | ctxkit --ollama gpt-oss:20b
```


## Copying Output

To copy the output of ctxkit and paste it into your favorite AI chat application, pipe ctxkit's
output into the clipboard tool for your platform.

**macOS**

```sh
ctxkit -m 'Hello!' | pbcopy
```

**Windows**

```sh
ctxkit -m 'Hello!' | clip
```

**Linux**

```sh
ctxkit -m 'Hello!' | xsel -ib
```


## Variables

You can specify one or more variable references in a message's text, a file path, a directory path,
or a URL using the syntax, `{{var}}`. A variable's value is specified using the `-v` argument. For
example:

```sh
ctxkit -v package ctxkit -m 'Write brief overview of the Python package, "{{package}}"'
```


## Configuration Files

ctxkit JSON configuration files allow you to construct complex prompts in one or more JSON files.


### Example: Write Unit Tests

To generate a prompt to write unit tests for a function or method in a module, create a
configuration file similar to the following:

```json
{
    "items": [
        {"message": "Write the unit test methods to cover the code in the {{scope}}."},
        {"file": "src/my_package/{{base}}.py"},
        {"file": "src/tests/test_{{base}}.py"}
    ]
}
```

In this example, the "scope" variable allows you to specify what you want to write unit tests for.
The "base" variable specifies the base sub-module name. To generate the prompt, run ctxkit:

```sh
ctxkit -v base main -v scope "main function" -c unittest.json
```

This outputs:

```
<system>
...
</system>

Write the unit test methods to cover the code in the main function.

<src/my_package/main.py>
# main.py
</src/my_package/main.py>

<src/tests/test_main.py>
# test_main.py
</src/tests/test_main.py>
```


## Usage

Using the `ctxkit` command line application, you can add any number of ordered *context items* of
the following types: configuration files (`-c`), messages (`-m`), file path or URL content (`-i` and
`-f`), and directories (`-d`).

```
usage: ctxkit [-h] [-g] [-o PATH] [-b] [-s PATH] [-c PATH] [-m TEXT] [-i PATH]
              [-t PATH] [-f PATH] [-d PATH] [-v VAR EXPR] [-x EXT] [-l INT]
              [--ollama MODEL | --grok MODEL] [--extract] [--temp NUM]
              [--topp NUM]

options:
  -h, --help           show this help message and exit
  -g, --config-help    display the JSON configuration file format
  -o, --output PATH    output to the file path
  -b, --backup         backup the output file with ".bak" extension
  -s, --system PATH    the system prompt file path, "" for no system prompt

Prompt Items:
  -c, --config PATH    process the JSON configuration file path or URL
  -m, --message TEXT   add a prompt message
  -i, --include PATH   add the file path or URL text
  -t, --template PATH  add the file path or URL template text
  -f, --file PATH      add the file path or URL as a text file
  -d, --dir PATH       add a directory's text files
  -v, --var VAR EXPR   define a variable (reference with "{{var}}")

Directory Options:
  -x, --ext EXT        add a directory text file extension
  -l, --depth INT      the maximum directory depth, default is 0 (infinite)

API Calling:
  --ollama MODEL       pass to the Ollama API
  --grok MODEL         pass to the Grok API
  --extract            extract response files - USE WITH CAUTION!
  --temp NUM           set the model response temperature
  --topp NUM           set the model response top_p
```


## Configuration File Format

The ctxkit `-g` argument outputs the JSON configuration file format defined using the
[Schema Markdown Language](https://craigahobbs.github.io/schema-markdown-js/language/).

```schema-markdown
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
```


## Development

This package is developed using [python-build](https://github.com/craigahobbs/python-build#readme).
It was started using [python-template](https://github.com/craigahobbs/python-template#readme) as follows:

```
template-specialize python-template/template/ ctxkit/ -k package ctxkit -k name 'Craig A. Hobbs' -k email 'craigahobbs@gmail.com' -k github 'craigahobbs' -k noapi 1
```
