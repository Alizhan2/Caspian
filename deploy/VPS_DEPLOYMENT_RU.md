# Постоянный deployment на VPS

Этот вариант работает 24/7 независимо от ноутбука. На Ubuntu 22.04/24.04 с Docker Compose:

```bash
git clone https://github.com/Alizhan2/Caspian.git
cd Caspian
cp deploy/.env.production.example .env.production
# заполните реальные секреты в .env.production
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
```

После запуска приложение доступно на порту 80 сервера. Для HTTPS и собственного домена перед портом 80 нужно подключить Caddy, Nginx или Cloudflare. Проверка:

```bash
curl http://SERVER_IP/api/health
docker compose --env-file .env.production -f docker-compose.prod.yml ps
```

Не публикуйте `.env.production`, ключи Copernicus, AIS или MinIO в GitHub.
