.PHONY: install test lint run demo eval clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=gridguard --cov-report=term-missing

lint:
	ruff check gridguard tests

run:
	uvicorn gridguard.main:app --reload --port 8000

demo:
	python -m gridguard.cli demo

eval:
	python -m gridguard.evaluation

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
