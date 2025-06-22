FROM node:16-alpine as build

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json .npmrc ./

# Install dependencies
RUN npm install --legacy-peer-deps

# Copy source code
COPY . .

# Build the app
RUN GENERATE_SOURCEMAP=false npm run build

# Production stage
FROM nginx:stable-alpine

# Copy built app to nginx serve directory
COPY --from=build /app/build /usr/share/nginx/html

# Add nginx configuration to handle SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
