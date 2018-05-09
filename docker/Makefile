 help:
	@cat Makefile

DOCKER_FILE=Dockerfile
SRC?=$(shell dirname `pwd`)
DATA?="$(SRC)/data"
ifeq ($(GPU),)
DOCKER=docker
else
DOCKER=NV_GPU=$(GPU) nvidia-docker
endif

build:
	docker build -t aes-lac-2018 -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it --ipc=host -v $(SRC):/workspace -v $(DATA):/workspace/data aes-lac-2018 bash

ipython: build
	$(DOCKER) run -it --ipc=host -v $(SRC):/workspace -v $(DATA):/workspace/data aes-lac-2018 ipython

notebook: build
	$(DOCKER) run -it --ipc=host -v $(SRC):/workspace -v $(DATA):/workspace/data --net=host aes-lac-2018
