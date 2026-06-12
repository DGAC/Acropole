.PHONY: install check test format lint audit ci clean docs-serve

install:  ## Install all dependencies
	uv sync --all-groups

check: lint audit test  ## Run all checks

lint:  ## Linter + type checker
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src tests

format:  ## Auto-format code
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:  ## Run tests with coverage
	uv run pytest

audit:  ## Security audit
	uv run pip-audit

ci: install check  ## Full CI pipeline

clean:  ## Clean artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml coverage_html dist htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

docs-serve:  ## Serve docs locally
	uv sync --group docs --quiet
	uv run mkdocs serve

help:  ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
