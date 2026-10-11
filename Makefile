PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install build-ext check-skeleton test docker bench-gemm bench-gemm-all bench-rmsnorm bench-softmax profile-gemm profile-rmsnorm profile-softmax clean

install:
	$(PIP) install -e ".[dev]" --no-build-isolation

build-ext:
	$(PYTHON) setup.py build_ext --inplace

check-skeleton:
	$(PYTHON) scripts/check_skeleton.py

test:
	$(PYTHON) -m pytest tests

docker:
	scripts/docker_run.sh "$${CUDA_KERNELS_IMAGE}"

bench-gemm:
	$(PYTHON) benchmarks/bench_gemm.py --impl naive --m 1024 --n 1024 --k 1024

bench-gemm-all:
	PYTHON=$(PYTHON) bash scripts/bench_gemm_all.sh

bench-rmsnorm:
	$(PYTHON) benchmarks/bench_rmsnorm.py --impl cuda --rows 4096 --hidden 4096

bench-softmax:
	$(PYTHON) benchmarks/bench_softmax.py --impl cuda --rows 4096 --cols 2048

profile-gemm:
	bash profiling/profile_gemm.sh thread_tile_8x8 4096 4096 4096

profile-rmsnorm:
	bash profiling/profile_rmsnorm.sh cuda 4096 4096

profile-softmax:
	bash profiling/profile_softmax.sh cuda 4096 2048

clean:
	rm -rf build dist *.egg-info python/cuda_ai_kernels/*.so
