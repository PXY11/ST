import asyncio
import time
from decimal import Decimal

import numpy as np
import talib as ta
from atom.model import *

from strategy_base.base import CommonStrategy

"""
cache:
{
    "pos": {"symbol":"-2"}
}
"""


class Strategy(CommonStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = self.config["strategy"]["params"]
        self.kline_size = self.params["kline_size"]
        self.kline_duration = getattr(KlineDuration, self.params["kline_duration"])
        self.cal_kline_timeout = self.params["kline_timeout"]
        self.open_qty = Decimal(self.params["open_qty"])
        self.cache = dict()
        self.kline_symbol = dict()  # s_kline --> s
        self.kline_use_float = True

    def update_config_before_init(self):
        self.custom_symbol_config_exchanges.add("binance")

    async def before_strategy_start(self):
        self.cache = await self.redis_get_cache(use_decimal=True)
        self.cache.setdefault("pos", dict())
        for s in self.symbol_list:
            if s not in self.cache["pos"]:
                self.cache["pos"][s] = 0
        await self.redis_set_cache(self.cache)
        self.logger.info(f"cache: {self.cache}")
        for s in self.symbol_list:
            cfg = self.get_symbol_config(s)
            s_kline = f"binance.{cfg['base']}_usdt.spot.na"
            self.kline_symbol[s_kline] = s
            await self.set_history_bar(s_kline, self.kline_duration, self.kline_size)
        self.subscribe_orderbook(self.symbol_list)

    def cal_ma_diff_one(self, s, bar):
        ma_period = self.params['maPeriod']
        am_close, am_volume = bar[4], bar[5]
        d_ema = ta.DEMA(am_close, ma_period)
        v_wap = ta.SUM(am_close * am_volume, ma_period) / ta.SUM(am_volume, ma_period)
        self.logger.info(f"{s}:dema&vwap:{d_ema[-1], v_wap[-1]}")
        ma_diff = d_ema - v_wap
        ma_conf = np.sign(ma_diff[-1])
        return ma_conf

    async def check_bar_arrive(self):
        start = time.time()
        last_completed_bar_ts = int(start // self.kline_duration.value * self.kline_duration.value - self.kline_duration.value)
        self.logger.info(f"start to check {last_completed_bar_ts} bar")
        valid_bars = dict()
        while time.time() - start < self.cal_kline_timeout:
            finished = True
            for s in self.kline_symbol:
                bar = self.get_history_bar(s, self.kline_duration, self.kline_size, bar_type=2, completed=True)
                if bar[0][-1] >= last_completed_bar_ts:
                    valid_bars[s] = bar
                else:
                    finished = False
            if not finished:
                await asyncio.sleep(1)
            else:
                break
        return valid_bars

    def calculate_ma_diff_by_bars(self, bars):
        ma_diff_mapping = dict()
        for s, bar in bars.items():
            ma_diff = self.cal_ma_diff_one(s, bar)
            ma_diff_mapping[s] = ma_diff
        return ma_diff_mapping

    def calculate_er_by_bars(self, bars):
        er_period = self.params['erPeriod']
        def erIndicator(am_close):
            change = np.abs(am_close[er_period:] - am_close[:-er_period])
            volatility = ta.SUM(np.abs(am_close[1:] - am_close[:-1]), er_period)
            er = (change[-1]) / volatility[-1]
            return er

        er_mapping = dict()
        for s, bar in bars.items():
            er_mapping[s] = erIndicator(bar[4])
        return max(er_mapping.items(), key=lambda item: item[1])

    def cal_signal(self, bars):
        max_er_symbol, max_er = self.calculate_er_by_bars(bars)  # s_kline, er
        ma_diff = self.calculate_ma_diff_by_bars(bars)  # s_kline, ma
        self.logger.info(f"cal result: max_er: {max_er_symbol} = {max_er}. ma_diff: {ma_diff}")
        exit_symbols = []
        for s_kline in bars.keys():  # s, pos
            s = self.kline_symbol[s_kline]
            v = Decimal(self.cache["pos"][s])
            if v == 0:
                continue
            if ma_diff[s_kline] < 0:
                exit_symbols.append(s)
        open_symbol = None
        if self.cache["pos"][self.kline_symbol[max_er_symbol]] == 0:
            if ma_diff[max_er_symbol] > 0:
                open_symbol = self.kline_symbol[max_er_symbol]
        return open_symbol, exit_symbols

    async def handle_open(self, s_open):
        if not s_open:
            return
        cfg = self.get_symbol_config(s_open)
        price = await self.get_latest_price(s_open)
        qty = self.open_qty / Decimal(cfg["contract_value"]) / price
        qty = max(cfg["min_quantity_val"], qty)
        qty_real = await self.simple_compulsory_order(s_open, qty, OrderSide.Buy, OrderPositionSide.Open, tag="s_open")
        if qty_real == 0:
            return
        self.cache["pos"][s_open] = qty_real
        await self.redis_set_cache(self.cache)

    async def handle_exit(self, s_exit):
        if not s_exit:
            return
        for s in s_exit:
            qty = self.cache["pos"][s]
            qty_real = await self.simple_compulsory_order(s, qty, OrderSide.Sell, OrderPositionSide.Close, tag="s_exit")
            self.cache["pos"][s] -= qty_real
        await self.redis_set_cache(self.cache)

    async def strategy_core(self):
        while True:
            now = int(time.time())
            next_ts = int(now // self.kline_duration.value * self.kline_duration.value + self.kline_duration.value)
            await asyncio.sleep(next_ts - now)
            self.cache = await self.redis_get_cache(use_decimal=True)
            bars = await self.check_bar_arrive()
            if len(bars) == 0:
                self.logger.error(f"no kline ready now")
                continue
            if len(bars) != len(self.symbol_list):
                self.logger.warning(f"not all bars arrive, got: {','.join(bars.keys())}")
            s_open, s_exit = self.cal_signal(bars)
            await self.handle_open(s_open)
            await self.handle_exit(s_exit)
            if s_open or s_exit:
                self.logger.info(f"process position finished. s_open={s_open} s_exit={s_exit} pos={self.cache['pos']}")


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    t = Strategy(loop)
    t.run_forever()
