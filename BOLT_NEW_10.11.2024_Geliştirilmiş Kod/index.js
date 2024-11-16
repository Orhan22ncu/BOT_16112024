require('dotenv').config();
import express,{ json,urlencoded } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import { setupDatabase } from './config/database';
import { setupRedis } from './config/redis';
import tradingRoutes from './routes/trading';
import errorHandler from './middleware/errorHandler';
import { info,error as _error } from './utils/logger';

const app = express();
const PORT = process.env.PORT || 3000;

// Security middleware
app.use(helmet());
app.use(cors());
app.use(compression());

// Body parser
app.use(json());
app.use(urlencoded({ extended: true }));

// Request logging
app.use((req, res, next) => {
  info(`${req.method} ${req.url}`);
  next();
});

// Database setup with retry mechanism
const connectWithRetry = async () => {
  try {
    await setupDatabase();
    await setupRedis();
  } catch (error) {
    _error('Connection failed, retrying in 5 seconds...');
    setTimeout(connectWithRetry, 5000);
  }
};

connectWithRetry();

// Routes
app.use('/api/trading', tradingRoutes);

// Error handling
app.use(errorHandler);

// Graceful shutdown
process.on('SIGTERM', () => {
  info('SIGTERM received. Shutting down gracefully...');
  server.close(() => {
    info('Process terminated');
  });
});

const server = app.listen(PORT, () => {
  info(`Server running on port ${PORT}`);
});