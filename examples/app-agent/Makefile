# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE


# The specs
SPEC ?= spec.md
UPDATE ?= update.md
APP ?= app


# The model to use
MODEL ?= --api $(if $(CLAUDE),claude claude-opus-4-1,grok grok-4-fast-reasoning)


# The ctxkit options
CTXKIT_ARGS := -b$(if $(TEST),, --extract)$(if $(TEST_PROMPT), --noapi, $(MODEL))$(if $(ARGS), $(ARGS))
CTXKIT_APP := -d $(APP)/ -x bare -x json


.phony: help
help:
	@echo 'usage: make [help|clean|superclean|start|create|improve|revert|review|run|update]'


.phony: clean
clean:
	rm -rf build/ $(APP)/


.phony: superclean
superclean: clean
	rm -rf spec.md update.md


build/venv.build:
	mkdir -p $(dir $@)
	python3 -m venv $(basename $@) --upgrade-deps
	$(basename $@)/bin/pip install ctxkit markdown-up
	touch $@


.phony: create
create: build/venv.build
	build/venv/bin/ctxkit $(CTXKIT_ARGS) -c prompts/create.json \
		-m 'Please create a MarkdownUp application in the "$(APP)/" directory. At a minimum, create two files:' \
		-m '    "$(APP)/$(APP).bare" - the MarkdownUp frontend application and main function. Do not run the main function.' \
		-m '    "$(APP)/$(APP).md" - the application Markdown file that runs the main function' \
		-m "Here's the application specification:" \
		-i $(SPEC)


.phony: improve
improve: build/venv.build
	build/venv/bin/ctxkit $(CTXKIT_ARGS) -c prompts/improve.json $(CTXKIT_APP)


.phony: lint
lint: build/venv.build
	while ! $(MAKE) lint-inner; do sleep 0; done


.phony: lint-inner
lint-inner: build/venv.build
	@echo 'Compiling $(APP)/$(APP).bare ...'
	@if ! build/venv/bin/bare -m $(APP)/$(APP).bare > build/compile.txt; then \
		echo 'Compile FAILED! Fixing...'; \
		build/venv/bin/ctxkit $(CTXKIT_ARGS) $(CTXKIT_APP) -f build/compile.txt -m 'Please fix the BareScrdipt execution issues.'; \
		false; \
	else \
		echo 'Linting $(APP)/*.bare ...'; \
		if ! build/venv/bin/bare -s $(APP)/*.bare > build/lint.txt; then \
			echo 'Lint FAILED! Fixing...'; \
			build/venv/bin/ctxkit $(CTXKIT_ARGS) $(CTXKIT_APP) -f build/lint.txt -m 'Please fix the static analysis issues.'; \
			false; \
		else \
			echo 'Success!'; \
		fi; \
	fi


.phony: review
review: build/venv.build
	build/venv/bin/ctxkit $(CTXKIT_ARGS) -c prompts/review.json $(CTXKIT_APP)


.phony: run
run: build/venv.build
	build/venv/bin/markdown-up $(APP)/$(APP).md


.phony: update
update: build/venv.build
	build/venv/bin/ctxkit $(CTXKIT_ARGS) -c prompts/update.json -i $(UPDATE) -c prompts/update2.json $(CTXKIT_APP)
