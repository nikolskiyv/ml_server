.PHONY: build
build:
	docker-compose -f docker-compose.yml build

.PHONY: run_all_docker
run_all_docker:
	docker-compose -f docker-compose.yml build && \
	docker-compose -f docker-compose.yml up app

.PHONY: stop_all_docker
stop_all_docker:
	docker-compose -f docker-compose.yml down

.PHONY: lint
lint:
	flake8 src && bandit -r src/app
