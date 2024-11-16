from binance.client import Client
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class MarketDataCollector:
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)
        
    def fetch_dual_market_data(self,
                             base_symbol: str = 'BTCUSDT',
                             target_symbol: str = 'BCHUSDT',
                             interval: str = '1m',
                             days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """BTC ve BCH verilerini çek"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # BTC verilerini çek
        btc_klines = self.client.get_historical_klines(
            base_symbol,
            interval,
            start_time.strftime("%d %b %Y %H:%M:%S"),
            end_time.strftime("%d %b %Y %H:%M:%S")
        )
        
        # BCH verilerini çek
        bch_klines = self.client.get_historical_klines(
            target_symbol,
            interval,
            start_time.strftime("%d %b %Y %H:%M:%S"),
            end_time.strftime("%d %b %Y %H:%M:%S")
        )
        
        # DataFrame'lere dönüştür
        btc_df = self._create_dataframe(btc_klines)
        bch_df = self._create_dataframe(bch_klines)
        
        # Zaman indekslerini senkronize et
        merged_df = pd.merge(
            btc_df,
            bch_df,
            left_index=True,
            right_index=True,
            suffixes=('_btc', '_bch')
        )
        
        btc_df = merged_df.filter(like='_btc')
        bch_df = merged_df.filter(like='_bch')
        
        return btc_df, bch_df
    
    def _create_dataframe(self, klines: List) -> pd.DataFrame:
        """Kline verilerini DataFrame'e dönüştür"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        # Veri tiplerini dönüştür
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df