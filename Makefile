install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	pytest -vv .

format:
	python -m black .

lint:
	python -m pylint --disable=R,C --exit-zero --recursive=y *

all: install format lint test
