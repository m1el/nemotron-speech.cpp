.PHONY: f
CPPFLAGS = -std=c++23 -Wextra -Wpedantic -Wall -O2
LDFLAGS = -lm

all: preprocessor

f:
	g++ preprocessor.cpp $(CPPFLAGS) $(LDFLAGS) -o preprocessor
	./preprocessor

preprocessor: preprocessor.cpp
	g++ preprocessor.cpp $(CPPFLAGS) $(LDFLAGS) -o preprocessor
	./preprocessor