.PHONY: all clean test

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -O2
CXXFLAGS += -I include
LDFLAGS = -lm

# Source files
SRCS = src/ggml_weights.cpp src/ops.cpp src/conv_subsampling.cpp src/conformer_modules.cpp src/conformer_encoder.cpp src/rnnt_decoder.cpp src/rnnt_joint.cpp src/greedy_decode.cpp src/tokenizer.cpp
OBJS = $(SRCS:.cpp=.o)

# Main targets
all: nemotron-speech preprocessor test_weights test_ops test_conv_subsampling test_conformer_modules test_conformer_encoder test_rnnt_decoder test_rnnt_joint test_greedy_decode test_tokenizer

# Main ASR executable
nemotron-speech: main.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Preprocessor (standalone)
preprocessor: preprocessor.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

# Test weight loading
test_weights: tests/test_weights.cpp src/ggml_weights.o
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test ops
test_ops: tests/test_ops.cpp src/ops.o
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test conv subsampling
test_conv_subsampling: tests/test_conv_subsampling.cpp src/conv_subsampling.o src/ops.o src/ggml_weights.o
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test conformer modules
test_conformer_modules: tests/test_conformer_modules.cpp src/conformer_modules.o src/ops.o src/ggml_weights.o
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test conformer encoder
test_conformer_encoder: tests/test_conformer_encoder.cpp src/conformer_encoder.o src/conformer_modules.o src/conv_subsampling.o src/ops.o src/ggml_weights.o
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test RNNT decoder
test_rnnt_decoder: tests/test_rnnt_decoder.cpp src/rnnt_decoder.o src/ops.o src/ggml_weights.o
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test RNNT joint
test_rnnt_joint: tests/test_rnnt_joint.cpp src/rnnt_joint.o src/ops.o src/ggml_weights.o
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test greedy decode
test_greedy_decode: tests/test_greedy_decode.cpp src/greedy_decode.o src/rnnt_joint.o src/rnnt_decoder.o src/conformer_encoder.o src/conformer_modules.o src/conv_subsampling.o src/ops.o src/ggml_weights.o
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test tokenizer
test_tokenizer: tests/test_tokenizer.cpp src/tokenizer.o
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Compile source files
src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run tests (skip slow encoder test by default)
test: test_weights test_ops test_conv_subsampling test_conformer_modules test_rnnt_decoder test_rnnt_joint test_tokenizer
	./test_weights weights/model.bin
	./test_ops
	./test_conv_subsampling
	./test_conformer_modules
	./test_rnnt_decoder
	./test_rnnt_joint
	./test_tokenizer

# Run all tests including slow ones
test_all: test test_conformer_encoder test_greedy_decode
	./test_conformer_encoder
	./test_greedy_decode

# Clean build files
clean:
	rm -f nemotron-speech preprocessor test_weights test_ops test_conv_subsampling test_conformer_modules test_conformer_encoder test_rnnt_decoder test_rnnt_joint test_greedy_decode test_tokenizer $(OBJS)

# Development helpers
f:
	$(CXX) preprocessor.cpp $(CXXFLAGS) $(LDFLAGS) -o preprocessor
	./preprocessor
