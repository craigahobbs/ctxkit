# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE


# The specs
SPEC ?= spec.md
UPDATE ?= update.md
APP ?= app.md


# The model to use
MODEL ?= $(if $(CLAUDE),--claude claude-opus-4-1,--grok grok-code-fast-1)


# The ctxkit options
CTXKIT_ARGS := -b$(if $(TEST),, --extract)$(if $(TEST_PROMPT),, $(MODEL))$(if $(ARGS), $(ARGS))


.phony: help
help:
	@echo 'usage: make [help|clean|superclean|start|create|improve|revert|review|run|update]'


.phony: clean
clean:
	rm -rf $(APP) $(APP).bak


.phony: superclean
superclean: clean
	rm -rf build/ spec.md update.md


build/venv.build:
	mkdir -p $(dir $@)
	python3 -m venv $(basename $@) --upgrade-deps
	$(basename $@)/bin/pip install ctxkit markdown-up
	touch $@


$(APP): build/venv.build
	build/venv/bin/ctxkit $(CTXKIT_ARGS) -c prompts/create.json -m 'The application filename is "$(APP)".' -i $(SPEC)


.phony: start
start:
	$(MAKE) --no-print-directory clean
	$(MAKE) --no-print-directory create
	$(MAKE) --no-print-directory review
	$(MAKE) --no-print-directory run


.phony: create
create: $(APP)


.phony: improve
improve: $(APP)
	build/venv/bin/ctxkit $(CTXKIT_ARGS) -c prompts/improve.json -f $(APP)


.phony: revert
revert: $(APP)
	if [ -f $(APP).bak ]; then mv $(APP).bak $(APP); fi


.phony: review
review: $(APP)
	build/venv/bin/ctxkit $(CTXKIT_ARGS) -c prompts/review.json -f $(APP)


.phony: run
run: $(APP)
	build/venv/bin/markdown-up $(APP)


.phony: update
update: $(APP)
	build/venv/bin/ctxkit $(CTXKIT_ARGS) -c prompts/update.json -i $(UPDATE) -c prompts/update2.json -f $(APP)
