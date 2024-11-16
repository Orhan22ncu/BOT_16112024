const fs = require('fs');
const { execSync } = require('child_process');
const logger = require('./utils/logger');

async function setup() {
  try {
    // Check Python installation
    execSync('python --version');
    logger.info('Python installation verified');

    // Install Python dependencies
    execSync('pip install -r requirements.txt');
    logger.info('Python dependencies installed');

    // Create necessary directories
    const dirs = ['logs', 'models', 'data'];
    dirs.forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir);
        logger.info(`Created directory: ${dir}`);
      }
    });

    // Initialize environment variables
    if (!fs.existsSync('.env')) {
      fs.copyFileSync('.env.example', '.env');
      logger.info('Created .env file');
    }

    logger.info('Setup completed successfully!');
  } catch (error) {
    logger.error('Setup failed:', error);
    process.exit(1);
  }
}

setup();