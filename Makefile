# =============================================================================
# Makefile for the MLOps Zoomcamp Repository
#
# Usage:
#   make <target> MODULE=<module_directory>
#
# Example:
#   make install MODULE=06-best-practices
#   make distclean MODULE=06-best-practices
# =============================================================================

# Default to Python 3
PYTHON = python3

# Check if MODULE variable is set for targets that require it
check_module = @bash -c 'if [ -z "$(MODULE)" ]; then echo "Error: MODULE variable is not set. Usage: make $(MAKECMDGOALS) MODULE=<module_name>"; exit 1; fi'

# Phony targets are not files
.PHONY: help install test lint format clean distclean

# Default target: show help
help:
	@echo "Usage: make <target> [MODULE=<module_directory>]"
	@echo ""
	@echo "Targets:"
	@echo "  help                  Show this help message."
	@echo "  install               Install dependencies from requirements.txt for a specific module."
	@echo "                        -> Requires MODULE. Example: make install MODULE=02-experiment-tracking"
	@echo "  test                  Run pytest for a specific module."
	@echo "                        -> Requires MODULE. Example: make test MODULE=06-best-practices"
	@echo "  lint                  Run flake8 linter on the entire project."
	@echo "  format                Format all Python files in the project using black."
	@echo "  clean                 Remove temporary Python files (__pycache__, .pyc)."
	@echo "  distclean             Remove the virtual environment and all temporary files from a module."
	@echo "                        -> DESTRUCTIVE. Requires MODULE. Example: make distclean MODULE=01-intro"
	@echo ""

# Target to install dependencies for a specific module
install:
	$(check_module)
	@echo "--> Installing dependencies for module: $(MODULE)..."
	@if [ -f "$(MODULE)/requirements.txt" ]; then \
		$(PYTHON) -m venv "$(MODULE)/venv" && \
		. "$(MODULE)/venv/bin/activate" && \
		pip install --upgrade pip && \
		pip install -r "$(MODULE)/requirements.txt"; \
	else \
		echo "Warning: requirements.txt not found in $(MODULE). Nothing to install."; \
	fi

# Target to run tests for a specific module
test:
	$(check_module)
	@echo "--> Running tests for module: $(MODULE)..."
	@if [ -d "$(MODULE)/tests" ]; then \
		pytest "$(MODULE)/tests/"; \
	else \
		echo "Warning: 'tests' directory not found in $(MODULE). No tests to run."; \
	fi

# Target to lint the entire project
lint:
	@echo "--> Running linter (flake8) using .flake8 config..."
	flake8 .

# Target to format the entire project
format:
	@echo "--> Formatting project code with black..."
	black .

# Target to clean up temporary python files
clean:
	@echo "--> Cleaning up temporary Python files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Cleanup complete."

# Target to completely clean a module's environment and temp files
distclean: clean
	$(check_module)
	@echo "--> Removing virtual environment from module: $(MODULE)..."
	@if [ -d "$(MODULE)/venv" ]; then \
		rm -rf "$(MODULE)/venv"; \
		echo "Virtual environment removed from $(MODULE)."; \
	else \
		echo "Warning: Virtual environment not found in $(MODULE). Nothing to remove."; \
	fi