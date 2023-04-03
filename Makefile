debug:
	FLASK_DEBUG=1 python3 -m  flask run --host=0.0.0.0

run:
	python3 -m  flask run --host=0.0.0.0

init:
	python3 -m venv venv
	. venv/bin/activate