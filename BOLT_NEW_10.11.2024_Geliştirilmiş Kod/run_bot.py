import logging
from test_environment import check_environment, test_model_predictions
from futures_trader import FuturesGridTrader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # API anahtarlarını güvenli bir şekilde yükle
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        API_KEY = config['api_key']
        API_SECRET = config['api_secret']
        
        # Sistem kontrollerini yap
        logger.info("🔄 Sistem kontrolleri yapılıyor...")
        if not check_environment():
            logger.error("❌ Sistem kontrolleri başarısız!")
            return
        
        # Model testini yap
        logger.info("🔄 Model testi yapılıyor...")
        if not test_model_predictions():
            logger.error("❌ Model testi başarısız!")
            return
        
        # Trading botu başlat
        logger.info("🚀 Trading bot başlatılıyor...")
        trader = FuturesGridTrader(
            api_key=API_KEY,
            api_secret=API_SECRET,
            symbol='BCHUSDT',
            leverage=20,
            initial_balance=1000,
            grid_levels=5,
            grid_spacing=0.005
        )
        
        # Botu çalıştır
        trader.run()
        
    except KeyboardInterrupt:
        logger.info("👋 Bot kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"❌ Beklenmeyen hata: {e}")
        raise

if __name__ == "__main__":
    main()