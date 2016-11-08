all:
	docker run --rm -v "${PWD}":/home -w /home -it rwongone/anaconda:1.0 bash

dev:
	docker run --rm -v "${PWD}":/home -p 8888:8888 -w /home -it rwongone/anaconda:1.0 bash -c "./start.sh; bash"

build:
	docker build -t "rwongone/anaconda:1.0" .
