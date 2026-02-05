stop:
	@docker compose stop
down:
	@docker compose down
build:
	@docker compose build

run: stop down build
	@docker compose up -d --remove-orphans tts

logs:
	@docker compose logs -f tts

trt: stop down build
	@docker compose up trt
