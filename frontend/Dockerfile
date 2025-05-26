FROM nginx:alpine

WORKDIR /usr/share/nginx/html

# Copy frontend files
COPY frontend/index.html .
COPY frontend/styles.css .
COPY frontend/script.js .

# Copy Nginx configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"] 