REPO_NAME := $(shell basename `git rev-parse --show-toplevel`)
DVC_REMOTE := ${GDRIVE_FOLDER}/${REPO_NAME}


.PHONY:test
test:
	pytest unittests/

.PHONY:install-hooks
install-hooks:
	precommit install

.PHONY:tensorboard
tensorboard:
	tensorboard --logdir=model_instances

.PHONY:dvc
dvc:
	dvc init
	dvc remote add --default gdrive ${DVC_REMOTE}

.PHONY: prefect
prefect:
	prefect server start --use-volume
