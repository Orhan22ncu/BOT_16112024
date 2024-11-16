import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import *
import time
from datetime import datetime, timedelta
from model import DeepLearningModel
from trading_agent import DQLAgent
from risk_management import RiskManager
from market_regime import MarketRegimeDetector
from feature_engineering import FeatureEngineer
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt'),
        logging.StreamHandler()
    ]
)

class FuturesGridTrader:
    def __init__(self, api_key, api_secret, symbol='BCHUSDT', leverage=20,
                 initial_balance=1000, grid_levels=5, grid_spacing=0.005):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.leverage = leverage
        self.initial_balance = initial_balance
        self.grid_levels = grid_levels
        self.grid_spacing = grid_spacing
        
        # Bileşenleri başlat
        self.risk_manager = RiskManager(max_position_size=0.2, max_drawdown=0.15)
        self.regime_detector = MarketRegimeDetector()
        self.feature_engineer = FeatureEngineer()
        
        # Model ve Agent'ı yükle
        self.load_models()
        
        # Grid pozisyonlarını takip et
        self.grid_positions = []
        self.active_orders = {}
        
        # Performans metrikleri
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'max_drawdown': 0
        }
    
    def load_models(self):
        try:
            self.model = DeepLearningModel.load_model('bch_model.h5')
            self.agent = DQLAgent.load('trading_agent.h5')
            logging.info("Modeller başarıyla yüklendi")
        except Exception as e:
            logging.error(f"Model yükleme hatası: {e}")
            raise
    
    def setup_futures_account(self):
        """Futures hesabını ayarla"""
        try:
            # Kaldıraç ayarla
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            
            # Marj tipi: ISOLATED
            self.client.futures_change_margin_type(symbol=self.symbol, marginType='ISOLATED')
            
            logging.info(f"Futures hesabı ayarlandı: {self.leverage}x kaldıraç")
        except Exception as e:
            logging.error(f"Futures hesap ayarlama hatası: {e}")
            raise
    
    def calculate_grid_levels(self, current_price):
        """Grid seviyelerini hesapla"""
        levels = []
        for i in range(-self.grid_levels, self.grid_levels + 1):
            level_price = current_price * (1 + i * self.grid_spacing)
            levels.append(level_price)
        return sorted(levels)
    
    def calculate_position_size(self, price):
        """Pozisyon büyüklüğünü hesapla"""
        account_info = self.client.futures_account()
        balance = float(account_info['totalWalletBalance'])
        
        # Risk yönetimi ile pozisyon büyüklüğü hesapla
        volatility = self.calculate_volatility()
        size = self.risk_manager.calculate_position_size(balance, volatility)
        
        # Kaldıraçlı işlem büyüklüğü
        leveraged_size = size * self.leverage
        
        return leveraged_size
    
    def calculate_volatility(self, window=20):
        """Volatilite hesapla"""
        klines = self.client.futures_klines(
            symbol=self.symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            limit=window
        )
        closes = pd.Series([float(k[4]) for k in klines])
        return closes.pct_change().std()
    
    def place_grid_orders(self, levels, position_size):
        """Grid emirlerini yerleştir"""
        for level in levels:
            try:
                # Long ve Short emirleri yerleştir
                long_order = self.client.futures_create_order(
                    symbol=self.symbol,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    side=SIDE_BUY,
                    price=level,
                    quantity=position_size
                )
                
                short_order = self.client.futures_create_order(
                    symbol=self.symbol,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    side=SIDE_SELL,
                    price=level,
                    quantity=position_size
                )
                
                self.active_orders[long_order['orderId']] = {
                    'price': level,
                    'side': 'LONG',
                    'size': position_size
                }
                self.active_orders[short_order['orderId']] = {
                    'price': level,
                    'side': 'SHORT',
                    'size': position_size
                }
                
                logging.info(f"Grid emirleri yerleştirildi: {level}")
            except Exception as e:
                logging.error(f"Grid emir yerleştirme hatası: {e}")
    
    def manage_positions(self, current_price, predictions):
        """Pozisyonları yönet"""
        try:
            # Açık pozisyonları kontrol et
            positions = self.client.futures_position_information(symbol=self.symbol)
            
            for position in positions:
                if float(position['positionAmt']) != 0:
                    entry_price = float(position['entryPrice'])
                    position_size = abs(float(position['positionAmt']))
                    unrealized_pnl = float(position['unRealizedProfit'])
                    
                    # Stop loss ve take profit kontrolü
                    if self.should_close_position(entry_price, current_price, 
                                               unrealized_pnl, predictions):
                        self.close_position(position)
                        
            # Yeni pozisyon açma kararı
            if self.should_open_position(current_price, predictions):
                self.open_new_position(current_price, predictions)
                
        except Exception as e:
            logging.error(f"Pozisyon yönetimi hatası: {e}")
    
    def should_close_position(self, entry_price, current_price, unrealized_pnl, predictions):
        """Pozisyon kapatma kararı"""
        # Stop loss kontrolü
        max_loss = entry_price * 0.02  # %2 stop loss
        if abs(current_price - entry_price) > max_loss:
            return True
        
        # Take profit kontrolü
        if unrealized_pnl > 0 and predictions['trend_reversal_probability'] > 0.7:
            return True
        
        return False
    
    def should_open_position(self, current_price, predictions):
        """Yeni pozisyon açma kararı"""
        # Piyasa rejimini kontrol et
        regime = self.regime_detector.detect_regime(predictions['features'])
        
        # Trend yönü ve güçlülük kontrolü
        trend_strength = predictions['trend_strength']
        trend_direction = predictions['trend_direction']
        
        # Risk durumu kontrolü
        if not self.risk_manager.should_trade(self.metrics['max_drawdown']):
            return False
        
        return (trend_strength > 0.7 and 
                regime[0] in [0, 1] and  # Trend veya Range rejimi
                predictions['entry_probability'] > 0.8)
    
    def run(self):
        """Trading botunu çalıştır"""
        self.setup_futures_account()
        
        while True:
            try:
                # Güncel veri al
                current_price = float(self.client.futures_symbol_ticker(symbol=self.symbol)['price'])
                
                # Özellik hesaplama ve tahmin
                features = self.feature_engineer.create_features(self.get_recent_data())
                predictions = self.make_predictions(features)
                
                # Grid seviyelerini güncelle
                levels = self.calculate_grid_levels(current_price)
                position_size = self.calculate_position_size(current_price)
                
                # Pozisyonları yönet
                self.manage_positions(current_price, predictions)
                
                # Grid emirlerini güncelle
                self.update_grid_orders(levels, position_size)
                
                # Metrikleri güncelle ve kaydet
                self.update_metrics()
                
                # Log
                self.log_status()
                
                # 1 dakika bekle
                time.sleep(60)
                
            except Exception as e:
                logging.error(f"Ana döngü hatası: {e}")
                time.sleep(60)
    
    def make_predictions(self, features):
        """Model tahminleri"""
        try:
            price_prediction = self.model.predict(features)
            action = self.agent.act(features, training=False)
            
            return {
                'next_price': price_prediction,
                'action': action,
                'features': features,
                'trend_strength': self.calculate_trend_strength(features),
                'trend_direction': self.calculate_trend_direction(features),
                'entry_probability': self.calculate_entry_probability(features),
                'trend_reversal_probability': self.calculate_reversal_probability(features)
            }
        except Exception as e:
            logging.error(f"Tahmin hatası: {e}")
            raise
    
    def calculate_trend_strength(self, features):
        """Trend güçlülük hesaplama"""
        return features['adx'].iloc[-1] / 100.0
    
    def calculate_trend_direction(self, features):
        """Trend yönü hesaplama"""
        return np.sign(features['ema_9'].iloc[-1] - features['ema_21'].iloc[-1])
    
    def calculate_entry_probability(self, features):
        """Giriş olasılığı hesaplama"""
        rsi = features['rsi'].iloc[-1]
        stoch_k = features['stoch_k'].iloc[-1]
        
        # RSI ve Stochastic değerlerini normalize et
        entry_prob = (50 - abs(rsi - 50)) / 50 * 0.5 + \
                    (50 - abs(stoch_k - 50)) / 50 * 0.5
        
        return entry_prob
    
    def calculate_reversal_probability(self, features):
        """Trend dönüş olasılığı hesaplama"""
        rsi = features['rsi'].iloc[-1]
        bb_width = features['bb_width'].iloc[-1]
        
        # Aşırı alım/satım ve volatilite bazlı olasılık
        if rsi > 70:
            return min(1.0, rsi / 100 + bb_width)
        elif rsi < 30:
            return min(1.0, (100 - rsi) / 100 + bb_width)
        else:
            return 0.0
    
    def update_metrics(self):
        """Performans metriklerini güncelle"""
        try:
            account = self.client.futures_account()
            current_balance = float(account['totalWalletBalance'])
            
            # Drawdown hesapla
            drawdown = (self.initial_balance - current_balance) / self.initial_balance
            self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)
            
            # Diğer metrikleri güncelle
            self.metrics['total_profit'] = current_balance - self.initial_balance
            
            # Metrikleri kaydet
            with open('trading_metrics.json', 'w') as f:
                json.dump(self.metrics, f)
                
        except Exception as e:
            logging.error(f"Metrik güncelleme hatası: {e}")
    
    def log_status(self):
        """Durum bilgilerini logla"""
        try:
            account = self.client.futures_account()
            positions = self.client.futures_position_information(symbol=self.symbol)
            
            logging.info(f"""
            ===== DURUM RAPORU =====
            Bakiye: {account['totalWalletBalance']} USDT
            Toplam Kar/Zarar: {self.metrics['total_profit']:.2f} USDT
            Max Drawdown: {self.metrics['max_drawdown']*100:.2f}%
            Açık Pozisyonlar: {len([p for p in positions if float(p['positionAmt']) != 0])}
            Aktif Grid Emirleri: {len(self.active_orders)}
            ========================
            """)
        except Exception as e:
            logging.error(f"Loglama hatası: {e}")

if __name__ == "__main__":
    API_KEY = "your_api_key"
    API_SECRET = "your_api_secret"
    
    trader = FuturesGridTrader(API_KEY, API_SECRET)
    trader.run()