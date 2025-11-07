# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE


# The specs
SPEC ?= spec.md
UPDATE ?= update.md
APP ?= app


# The model to use
MODEL ?= --api \
  $(if $(SMARTER), claude claude-sonnet-4-5, \
  $(if $(SMART), grok grok-code-fast, \
  grok grok-4-fast-reasoning))


# The ctxkit options
CTXKIT_ARGS := -b$(if $(TEST),, --extract)$(if $(TEST_PROMPT), --noapi, $(MODEL))$(if $(ARGS), $(ARGS))
CTXKIT_APP := -d $(APP)/ -x bare -x json -x md


.phony: help
help:
	@echo 'usage: make [help|clean|superclean|create|improve|revert|review|run|update]'


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
	build/venv/bin/ctxkit $(CTXKIT_ARGS) \
		-v app "$(APP)" \
		-c prompts/create.json \
		-i $(SPEC)


.phony: improve
improve: build/venv.build
	build/venv/bin/ctxkit $(CTXKIT_APP) $(CTXKIT_ARGS) \
		-c prompts/improve.json


.phony: lint
lint: build/venv.build
	while ! $(MAKE) lint-inner; do sleep 0; done


.phony: lint-inner
lint-inner: build/venv.build
	@echo 'Linting $(APP)/*.bare ...'
	if ! build/venv/bin/bare -d -m -x $(APP)/*.bare > build/lint.txt; then \
		echo 'Lint FAILED! Fixing...'; \
		build/venv/bin/ctxkit $(CTXKIT_ARGS) $(CTXKIT_APP) -f build/lint.txt -m 'Please fix the static analysis issues.'; \
		false; \
	else \
		echo 'Success!'; \
	fi


.phony: review
review: build/venv.build
	build/venv/bin/ctxkit $(CTXKIT_APP) $(CTXKIT_ARGS) \
		-c prompts/review.json


.phony: run
run: build/venv.build
	build/venv/bin/markdown-up $(APP)/$(APP).md


.phony: update
update: build/venv.build
	build/venv/bin/ctxkit $(CTXKIT_APP) $(CTXKIT_ARGS) \
		-c prompts/update.json \
		-i $(UPDATE) \
		-c prompts/update2.json
