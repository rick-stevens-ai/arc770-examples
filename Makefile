.PHONY: all clean rust julia check-env

check-env:
	@if [ ! -f "setup_env.sh" ]; then \
		echo "Error: setup_env.sh not found!"; \
		exit 1; \
	fi
	@source ./setup_env.sh; \
	if [ ! "$$ONEAPI_ENV_INITIALIZED" = "1" ]; then \
		echo "Error: oneAPI environment not properly initialized!"; \
		exit 1; \
	fi

all: check-env rust julia

rust: check-env
	$(MAKE) -C rust

julia: check-env
	$(MAKE) -C julia

clean:
	$(MAKE) -C rust clean
	$(MAKE) -C julia clean

test: check-env
	$(MAKE) -C rust test
	$(MAKE) -C julia test

.PHONY: test
