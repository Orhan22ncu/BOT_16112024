const { TradingBot } = require('../model/trading_bot');

let tradingBot = null;

const startTrading = async (req, res) => {
  try {
    if (!tradingBot) {
      tradingBot = new TradingBot();
      await tradingBot.start();
      res.json({ message: 'Trading started successfully' });
    } else {
      res.status(400).json({ message: 'Trading already running' });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

const stopTrading = async (req, res) => {
  try {
    if (tradingBot) {
      await tradingBot.stop();
      tradingBot = null;
      res.json({ message: 'Trading stopped successfully' });
    } else {
      res.status(400).json({ message: 'Trading not running' });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

const getStatus = async (req, res) => {
  try {
    if (tradingBot) {
      const status = await tradingBot.getStatus();
      res.json(status);
    } else {
      res.json({ status: 'stopped' });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

module.exports = {
  startTrading,
  stopTrading,
  getStatus
};