.PHONY: all clean test_all

all: test_all

test_all: check_device run_cg run_ttest run_matrix_multiply run_block_cholesky

check_device:
	julia --project=. check_device.jl

run_cg:
	cd conjugate_gradient && julia --project=. conjugate_gradient.jl

run_ttest:
	cd ttest && julia --project=. ttest_benchmark.jl

run_matrix_multiply:
	cd matrix_multiply && julia --project=. matrix_multiply_benchmark.jl

run_block_cholesky:
	cd block_cholesky && julia --project=. ct_reconstruction.jl

clean:
	find . -name "Manifest.toml" -type f -delete
