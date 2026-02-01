.PHONY: install sync clean

# pufferlib requires gcc (not clang) due to gcc-specific compiler flags
export CC=gcc
export CXX=g++

install: sync

sync:
	uv sync

clean:
	rm -rf .venv
	uv cache clean pufferlib
