const Redis = require('redis');

const setupRedis = async () => {
  const client = Redis.createClient({
    url: process.env.REDIS_URL
  });

  client.on('error', (err) => console.error('Redis Client Error', err));
  client.on('connect', () => console.log('Connected to Redis'));

  await client.connect();
  return client;
};

module.exports = { setupRedis };