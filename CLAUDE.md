# CLAUDE.md

Notes for coding agents working in this repository.

## Project

`ctxkit` is a Python CLI that builds AI prompts by concatenating ordered "context items" (messages, files, directories, URL content, included configs), optionally calls a provider API (Claude / Gemini / GPT / Grok / Ollama), and can extract response-emitted files back to disk (whole files or unified diffs).

## python-build

This is a [python-build](https://github.com/craigahobbs/python-build#readme) package. Read the python-build skill before running tests, lint, coverage, or changing the Makefile: [`../python-build/SKILL.md`](../python-build/SKILL.md) if that file exists, otherwise [https://raw.githubusercontent.com/craigahobbs/python-build/main/SKILL.md](https://raw.githubusercontent.com/craigahobbs/python-build/main/SKILL.md).

Local Makefile overrides:

- `PYLINT_ARGS` — missing docstring checks disabled
- no `SPHINX_DOC` — `make doc` is a no-op

`make clean` also cleans `examples/app-agent/`.

## Architecture

Entry point `ctxkit.main:main` (`src/ctxkit/main.py`) is the only non-trivial top-level flow. It:

1. Prepends `CTXKIT_FLAGS` env var to argv (used for user defaults like `--api grok grok-4-fast-reasoning`).
2. Parses argparse args — prompt-item flags (`-c -m -i -t -f -d -v`) all share `dest='items'` via the custom `TypedItemAction`, preserving **command-line order** as a list of `(item_type, value)` tuples. Item order matters: it is the order they appear in the final prompt.
3. Translates that list into a `{'items': [...]}` config dict matching the `CtxKitConfig` schema (defined as Schema Markdown in `config.py:CTXKIT_SMD`, parsed with `schema_markdown`). The schema is the source of truth — `-g/--config-help` prints it, and `-c` configs are validated against it.
4. Delegates to `config.process_config` / `process_config_items` (generator) to render each item to a string, then either prints, writes to `-o`, or pipes to `output_api_call`.
   - `--api` with no prompt items reads the prompt from stdin.
   - Without `--api`, stdout output starts with a `<system>…</system>` item; `-o` output omits the system prompt.
   - Every exception is printed as `\nError: …` to stderr with exit status 2.
   - `CTXKIT_FLAGS` is split on whitespace only (no quoting).

`long` (a list of lines) is a config-file-only item type with no CLI flag.

Key cross-cutting behaviors in `config.py`:

- **Variables** (`{{name}}`) — expanded in messages, `long` items, item paths, and `template` text, but **not** `include` text (that is the difference between the two). Unknown names expand to an empty string. The `variables` dict is mutated in order and shared with nested configs, so a `var` item must appear *before* the items that reference it.
- **`config` items recurse** into `process_config_items` with `root_dir` set to the included config's directory; relative paths in a nested config resolve against *that* config's location, not the cwd.
- **`is_url`** (regex `^[a-z]+:`) tells URLs from paths in item path resolution, `fetch_text` (the unified file-or-URL reader), and extraction (which skips URL-like tag names).
- **`--diff` mode** prepends `lineno:` to every line of `file` and `dir` items (`_add_line_numbers`), and, unless `-s` is given, `main.py` swaps the system prompt to `DEFAULT_SYSTEM_DIFF` (in `api/__init__.py`; instructs the model to reply with unified diffs). On extraction, `diff.apply_diff` reconstructs files.

API layer (`src/ctxkit/api/`):

- `api/__init__.py` registers providers in `API_PROVIDERS` (`claude`, `gemini`, `gpt`, `grok`, `ollama`). Each provider module exports a `<name>_chat(pool_manager, model, system_prompt, prompt, temperature=None, top_p=None, max_tokens=None)` generator (yields response chunks for streaming) and a `<name>_list(pool_manager)` function (returns available model names). To add a provider: add `api/<name>.py` with those two functions (reuse `_sse.iter_sse_events` for SSE streams; read the API key from an env var and raise `urllib3.exceptions.HTTPError` if unset), register `{'description', 'chat', 'list'}` in `API_PROVIDERS` (`description` feeds the `--help` epilog), add `src/tests/test_api_<name>.py`, and update the README.
- `output_api_call` streams chunks to the output, accumulates them, then runs `_extract_files` if `--extract` was passed.
- `_extract_files` parses the streamed response with `_R_FILENAME_TAG` (regex matching `<path>\n...\n</path>` blocks at line start). Content equal to `ctxkit: delete` deletes the file. Otherwise it writes (or, with `--diff`, applies a unified diff against the existing file).
- The default system prompts (`DEFAULT_SYSTEM`, `DEFAULT_SYSTEM_DIFF`) are what teach the model the `<filename>...</filename>` extraction protocol — changing the regex or the prompts requires updating both in lockstep.

## Conventions

- Single-package layout under `src/ctxkit/`; tests live in `src/tests/`. `test_main.py` covers `main.py`, `config.py`, and file extraction end to end via `main([...])`; `test_diff.py` tests `diff.py` directly; each provider has `test_api_<provider>.py`. Tests clear `os.environ` (so `CTXKIT_FLAGS` and API keys don't leak in), patch `sys.stdout` / `sys.stderr`, mock `urllib3.PoolManager`, and pass `-s ''` to omit the default system prompt.
- Existing coverage pragmas: `no cover` on the Python-version check, `no branch` on the dirname guards. Don't add new ones; write tests instead.
- `README.md` copies the `--help` usage block and `CTXKIT_SMD` verbatim (Usage and Configuration File Format sections). Update them when changing CLI flags, provider descriptions, or the schema.
