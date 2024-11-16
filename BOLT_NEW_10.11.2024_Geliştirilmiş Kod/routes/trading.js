const express = require('express');
const router = express.Router();
const { startTrading, stopTrading, getStatus } = require('../controllers/trading');

router.post('/start', startTrading);
router.post('/stop', stopTrading);
router.get('/status', getStatus);

module.exports = router;