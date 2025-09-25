CXX = icpx
CXXFLAGS = -fsycl

gpu_test: gpu_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f gpu_test
