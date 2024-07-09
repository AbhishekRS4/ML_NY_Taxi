install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	pytest -vv .

format:
	python -m black *.py

lint:
	python -m pylint --disable=R,C *.py

all: install format lint test