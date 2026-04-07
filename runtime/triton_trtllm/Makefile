stop:
	@docker compose -f docker-compose.cosyvoice2.unet.yml stop
down:
	@docker compose -f docker-compose.cosyvoice2.unet.yml down
build:
	@docker compose -f docker-compose.cosyvoice2.unet.yml build

run: stop down build
	@docker compose -f docker-compose.cosyvoice2.unet.yml up -d --remove-orphans tts

logs:
	@docker compose -f docker-compose.cosyvoice2.unet.yml logs -f tts

trt: stop down build
	@docker compose -f docker-compose.cosyvoice2.unet.yml up trt
