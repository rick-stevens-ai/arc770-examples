PROJECTS := matrix_multiply monte_carlo binomial_options

.PHONY: all clean test run $(PROJECTS)

all: $(PROJECTS)

$(PROJECTS):
	@echo "Building $@..."
	cd $@ && cargo build --release

run: $(PROJECTS)
	@for dir in $(PROJECTS); do \
		echo "\nRunning $$dir..."; \
		(cd $$dir && cargo run --release); \
	done

test: $(PROJECTS)
	@for dir in $(PROJECTS); do \
		echo "\nTesting $$dir..."; \
		(cd $$dir && cargo test --release); \
	done

clean:
	@for dir in $(PROJECTS); do \
		echo "Cleaning $$dir..."; \
		(cd $$dir && cargo clean); \
	done
