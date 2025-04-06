FROM node:18-slim

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install --only=production

# Copy source files
COPY . .

# Expose port
EXPOSE 3000

# Start the server
CMD ["node", "server.js"]