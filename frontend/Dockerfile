# MovieHarbor frontend: the static web UI, served by nginx, which proxies /api/ to the backend.
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html app.js styles.css /usr/share/nginx/html/
