# Compiler and flags
CXX = icpx
CXXFLAGS = -fsycl -O3 -std=c++17
GPU_FLAGS = -DSYCL_DEVICE_FILTER=level_zero:gpu -DONEAPI_DEVICE_SELECTOR="level_zero:gpu"
MKL_COPTS = -DMKL_ILP64 -qmkl=sequential
MKL_LIBS = -lsycl -lOpenCL -lpthread -lm -ldl
EXTRA_FLAGS = -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations

# Examples to build (removed block_cholesky)
EXAMPLES = matrix_mul_mkl black_scholes binomial monte_carlo_pi \
          student_t_test sparse_conjugate_gradient fourier_correlation \
          block_lu computed_tomography monte_carlo_european random_sampling

.PHONY: all clean run $(EXAMPLES) build_all run_all

all: build_all run_all

build_all: build_matrix_mul build_black_scholes build_binomial build_monte_carlo \
          build_student_t build_sparse_cg build_fourier build_block_lu \
          build_computed_tomography build_monte_carlo_european build_random_sampling

run_all: run_matrix_mul run_black_scholes run_binomial run_monte_carlo \
        run_student_t run_sparse_cg run_fourier run_block_lu \
        run_computed_tomography run_monte_carlo_european run_random_sampling

# Matrix Multiplication
build_matrix_mul:
	@echo "Building matrix_mul_mkl..."
	$(MAKE) -C matrix_mul_mkl clean
	$(MAKE) -C matrix_mul_mkl build

run_matrix_mul:
	@echo "Running matrix_mul_mkl (single precision)..."
	cd matrix_mul_mkl && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./matrix_mul_mkl single

# Black Scholes
build_black_scholes:
	@echo "Building black_scholes..."
	cd black_scholes && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) src/*.cpp -o black_scholes $(MKL_LIBS)

run_black_scholes:
	@echo "Running black_scholes..."
	cd black_scholes && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./black_scholes

# Binomial
build_binomial:
	@echo "Building binomial..."
	cd binomial && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) src/*.cpp -o binomial $(MKL_LIBS)

run_binomial:
	@echo "Running binomial..."
	cd binomial && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./binomial

# Monte Carlo Pi
build_monte_carlo:
	@echo "Building monte_carlo_pi..."
	cd monte_carlo_pi && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) mc_pi.cpp -o mc_pi $(MKL_LIBS)

run_monte_carlo:
	@echo "Running monte_carlo_pi..."
	cd monte_carlo_pi && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./mc_pi

# Student T-Test
build_student_t:
	@echo "Building student_t_test..."
	cd student_t_test && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) $(EXTRA_FLAGS) t_test.cpp -o t_test $(MKL_LIBS)
	cd student_t_test && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) $(EXTRA_FLAGS) t_test_usm.cpp -o t_test_usm $(MKL_LIBS)

run_student_t:
	@echo "Running student_t_test..."
	cd student_t_test && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./t_test
	cd student_t_test && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./t_test_usm

# Sparse Conjugate Gradient
build_sparse_cg:
	@echo "Building sparse_conjugate_gradient..."
	cd sparse_conjugate_gradient && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) $(EXTRA_FLAGS) sparse_cg.cpp -o sparse_cg $(MKL_LIBS)

run_sparse_cg:
	@echo "Running sparse_conjugate_gradient..."
	cd sparse_conjugate_gradient && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./sparse_cg

# Fourier Correlation
build_fourier:
	@echo "Building fourier_correlation..."
	cd fourier_correlation && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) fcorr_1d_buffers.cpp -o fcorr_1d_buff $(MKL_LIBS)
	cd fourier_correlation && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) fcorr_1d_usm.cpp -o fcorr_1d_usm $(MKL_LIBS)
	cd fourier_correlation && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) fcorr_2d_usm.cpp -o fcorr_2d_usm $(MKL_LIBS)

run_fourier:
	@echo "Running fourier_correlation..."
	cd fourier_correlation && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./fcorr_1d_buff 4096
	cd fourier_correlation && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./fcorr_1d_usm 4096
	cd fourier_correlation && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./fcorr_2d_usm

# Block LU Decomposition
build_block_lu:
	@echo "Building block_lu_decomposition..."
	cd block_lu_decomposition && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) $(EXTRA_FLAGS) \
		factor.cpp sgeblttrf.cpp auxi.cpp -o factor $(MKL_LIBS)
	cd block_lu_decomposition && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) $(EXTRA_FLAGS) \
		solve.cpp sgeblttrf.cpp sgeblttrs.cpp auxi.cpp -o solve $(MKL_LIBS)

run_block_lu:
	@echo "Running block_lu_decomposition..."
	cd block_lu_decomposition && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./factor
	cd block_lu_decomposition && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./solve

# Computed Tomography
build_computed_tomography:
	@echo "Building computed_tomography..."
	cd computed_tomography && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) $(EXTRA_FLAGS) \
		computed_tomography.cpp -o computed_tomography $(MKL_LIBS)

run_computed_tomography:
	@echo "Running computed_tomography..."
	cd computed_tomography && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" \
		./computed_tomography 400 400 input.bmp radon.bmp restored.bmp

# Monte Carlo European Options
build_monte_carlo_european:
	@echo "Building monte_carlo_european_opt..."
	cd monte_carlo_european_opt && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) $(EXTRA_FLAGS) \
		src/montecarlo_main.cpp -o montecarlo $(MKL_LIBS)

run_monte_carlo_european:
	@echo "Running monte_carlo_european_opt..."
	cd monte_carlo_european_opt && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./montecarlo

# Random Sampling Without Replacement
build_random_sampling:
	@echo "Building random_sampling_without_replacement..."
	cd random_sampling_without_replacement && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) $(EXTRA_FLAGS) \
		lottery.cpp -o lottery $(MKL_LIBS)
	cd random_sampling_without_replacement && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) $(EXTRA_FLAGS) \
		lottery_usm.cpp -o lottery_usm $(MKL_LIBS)
	cd random_sampling_without_replacement && $(CXX) $(CXXFLAGS) $(GPU_FLAGS) $(MKL_COPTS) $(EXTRA_FLAGS) \
		lottery_device_api.cpp -o lottery_device_api $(MKL_LIBS)

run_random_sampling:
	@echo "Running random_sampling_without_replacement..."
	cd random_sampling_without_replacement && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./lottery
	cd random_sampling_without_replacement && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./lottery_usm
	cd random_sampling_without_replacement && SYCL_DEVICE_FILTER=level_zero:gpu ONEAPI_DEVICE_SELECTOR="level_zero:gpu" ./lottery_device_api

clean:
	$(MAKE) -C matrix_mul_mkl clean
	rm -f black_scholes/black_scholes
	rm -f binomial/binomial
	rm -f monte_carlo_pi/mc_pi
	rm -f student_t_test/t_test student_t_test/t_test_usm
	rm -f sparse_conjugate_gradient/sparse_cg sparse_conjugate_gradient/genxir
	rm -f fourier_correlation/fcorr_1d_buff fourier_correlation/fcorr_1d_usm fourier_correlation/fcorr_2d_usm
	rm -f block_lu_decomposition/factor block_lu_decomposition/solve block_lu_decomposition/genxir
	rm -f computed_tomography/computed_tomography computed_tomography/radon.bmp computed_tomography/restored.bmp
	rm -f monte_carlo_european_opt/montecarlo
	rm -f random_sampling_without_replacement/lottery random_sampling_without_replacement/lottery_usm \
		random_sampling_without_replacement/lottery_device_api
