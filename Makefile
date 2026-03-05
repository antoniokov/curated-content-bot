-include .env.dev

ifndef SERVER
  $(error SERVER is not set. Create .env.dev with SERVER=<your-server-ip>)
endif

.PHONY: deploy deploy-rebuild test

deploy:
	ssh root@$(SERVER) 'cd /opt/curated-content-bot && git pull && systemctl restart curated-content-bot'

deploy-rebuild:
	ssh root@$(SERVER) 'cd /opt/curated-content-bot && git pull && systemctl restart curated-content-bot && systemctl start daily-refresh'

test:
	python3 -m pytest tests/test_bot.py -v
