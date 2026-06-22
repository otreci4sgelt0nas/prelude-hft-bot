"""
Paper Trading Engine — simulates order execution without real API calls.

Persists all virtual trades to paper_trades.db (SQLite) and calculates
realistic fill prices by walking real-time order-book depth from the
Polymarket CLOB public endpoint.

SQLite schema
─────────────
paper_trades
  id              INTEGER PK AUTOINCREMENT
  timestamp       TEXT    (YYYY-MM-DD HH:MM:SS)
  market_id       TEXT    (event slug)
  side            TEXT    (UP / DOWN)
  price           REAL    (simulated fill price)
  size            REAL    (number of shares)
  amount_usd      REAL    (notional in USD)
  status          TEXT    (open / closed)
  exit_price      REAL    (NULL while open)
  exit_timestamp  TEXT    (NULL while open)
  pnl             REAL    (NULL while open)
  reason          TEXT    (close reason)

paper_balance
  id        INTEGER PK AUTOINCREMENT
  timestamp TEXT
  balance   REAL
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
CLOB = "https://clob.polymarket.com"
DB_PATH = Path(__file__).parent.parent / "paper_trades.db"

MIN_SHARES     = 5
MAX_TOKEN_PRICE = 0.99
MIN_TOKEN_PRICE = 0.01

# Spread-based fallback impact model
IMPACT_SCALE = 100.0    # USD order size at which impact = 50 % of spread

_session = requests.Session()


# ── PaperEngine ──────────────────────────────────────────────────────────────

class PaperEngine:
    """
    Drop-in replacement for live trade execution when TRADING_MODE=PAPER.

    Key public API (mirrors trade_executor.py conventions):
      execute_buy(direction, amount_usd, token_up, token_down, market_id, get_price)
          -> (position_dict | None, message_str)

      close_position(direction, shares, entry_price, token_up, token_down,
                     market_id, get_price, reason)
          -> (exit_price, pnl, message_str)

      close_all(positions, token_up, token_down, market_id, get_price,
                trade_logger, reason, session_pnl, trade_history)
          -> (total_pnl, count, session_pnl, pnl_list)

      calculate_unrealized_pnl(positions, token_up, token_down, get_price)
          -> float

      get_portfolio_summary()
          -> dict
    """

    def __init__(self, db_path: str | None = None,
                 starting_balance: float = 1000.0):
        self.db_path = str(db_path or DB_PATH)
        self._starting_balance = starting_balance
        self._init_db()
        saved = self._load_balance()
        self.virtual_balance: float = saved if saved is not None else starting_balance

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp      TEXT    NOT NULL,
                    market_id      TEXT    NOT NULL,
                    side           TEXT    NOT NULL,
                    price          REAL    NOT NULL,
                    size           REAL    NOT NULL,
                    amount_usd     REAL    NOT NULL,
                    status         TEXT    NOT NULL DEFAULT 'open',
                    exit_price     REAL,
                    exit_timestamp TEXT,
                    pnl            REAL,
                    reason         TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_balance (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    balance   REAL NOT NULL
                )
            """)
            conn.commit()

    def _load_balance(self) -> float | None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT balance FROM paper_balance ORDER BY id DESC LIMIT 1"
                ).fetchone()
                return float(row[0]) if row else None
        except sqlite3.Error as e:
            logger.debug("Error loading paper balance: %s", e)
            return None

    def _save_balance(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO paper_balance (timestamp, balance) VALUES (?, ?)",
                    (now, self.virtual_balance),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.debug("Error saving paper balance: %s", e)

    def reset_balance(self, new_balance: float | None = None) -> None:
        """Reset virtual balance to starting amount (or a custom value)."""
        self.virtual_balance = new_balance if new_balance is not None else self._starting_balance
        self._save_balance()

    # ── Order-book depth fetcher ──────────────────────────────────────────────

    def _fetch_orderbook(self, token_id: str) -> dict:
        """Fetch live order book from Polymarket CLOB (public endpoint)."""
        try:
            resp = _session.get(
                f"{CLOB}/book",
                params={"token_id": token_id},
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.debug("Order-book fetch error token=%s: %s", token_id[:8], e)
        return {}

    # ── Fill-price simulation ─────────────────────────────────────────────────

    def simulate_fill_price(self, token_id: str, side: str,
                             amount_usd: float, get_price) -> float:
        """
        Calculate a realistic simulated fill price.

        Attempts to walk the live order-book depth first:
          BUY  → consumes ask levels cheapest → most expensive (VWAP)
          SELL → consumes bid levels most expensive → cheapest (VWAP)

        Falls back to a spread-based market-impact model if the book is
        unavailable or has insufficient liquidity.

        Args:
            token_id:   Polymarket CLOB token ID
            side:       "BUY" or "SELL"
            amount_usd: notional value of the order in USD
            get_price:  callable(token_id, side) -> float

        Returns:
            Simulated fill price clamped to [MIN_TOKEN_PRICE, MAX_TOKEN_PRICE].
        """
        book = self._fetch_orderbook(token_id)

        if side == "BUY":
            levels = sorted(
                book.get("asks", []),
                key=lambda x: float(x.get("price", 1.0)),
            )
        else:
            levels = sorted(
                book.get("bids", []),
                key=lambda x: float(x.get("price", 0.0)),
                reverse=True,
            )

        if levels:
            remaining_usd = amount_usd
            total_shares  = 0.0
            total_cost    = 0.0

            for level in levels:
                lp = float(level.get("price", 0))
                ls = float(level.get("size",  0))
                if lp <= 0 or ls <= 0:
                    continue
                available_usd = lp * ls
                fill_usd    = min(remaining_usd, available_usd)
                fill_shares = fill_usd / lp
                total_shares += fill_shares
                total_cost   += fill_usd
                remaining_usd -= fill_usd
                if remaining_usd <= 0.0:
                    break

            if total_shares > 0.0:
                vwap = total_cost / total_shares
                return max(MIN_TOKEN_PRICE, min(MAX_TOKEN_PRICE, vwap))

        # ── Spread-based fallback ──────────────────────────────────────────
        ask = get_price(token_id, "BUY")
        bid = get_price(token_id, "SELL")
        if ask <= 0:
            return 0.0
        spread = max(ask - bid, 0.001)
        # Impact: order that fills IMPACT_SCALE USD incurs 50 % of spread
        impact = spread * min(amount_usd / IMPACT_SCALE, 0.5)
        if side == "BUY":
            return min(ask + impact, MAX_TOKEN_PRICE)
        else:
            return max(bid - impact, MIN_TOKEN_PRICE)

    # ── Trade execution ───────────────────────────────────────────────────────

    def execute_buy(self, direction: str, amount_usd: float,
                    token_up: str, token_down: str,
                    market_id: str, get_price) -> tuple[dict | None, str]:
        """
        Simulate a market-buy order and persist it to SQLite.

        Returns:
            (position_dict, success_message) on success.
            (None,          error_message)   on failure.
        """
        token_id   = token_up   if direction == "up" else token_down
        side_label = "UP"       if direction == "up" else "DOWN"

        fill_price = self.simulate_fill_price(token_id, "BUY", amount_usd, get_price)
        if fill_price <= 0:
            return None, "✗ [PAPER] Cannot determine fill price"

        shares = round(amount_usd / fill_price, 2)
        if shares < MIN_SHARES:
            return None, (f"✗ [PAPER] Minimum {MIN_SHARES} shares required "
                          f"— increase trade amount")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO paper_trades
                        (timestamp, market_id, side, price, size, amount_usd, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'open')
                """, (now, market_id, side_label, fill_price, shares, amount_usd))
                conn.commit()
        except sqlite3.Error as e:
            logger.debug("Error inserting paper trade: %s", e)

        self.virtual_balance -= fill_price * shares
        self._save_balance()

        msg = (f"[PAPER] ✓ BUY {side_label} | "
               f"{shares:.2f}sh @ ${fill_price:.4f} = ${shares * fill_price:.2f}")
        return {
            'direction': direction,
            'price':     fill_price,
            'shares':    shares,
            'time':      datetime.now().strftime("%H:%M:%S"),
        }, msg

    def close_position(self, direction: str, shares: float, entry_price: float,
                       token_up: str, token_down: str,
                       market_id: str, get_price,
                       reason: str = "manual") -> tuple[float, float, str]:
        """
        Simulate closing a single open position.

        Returns:
            (exit_price, pnl, message)
        """
        token_id   = token_up if direction == "up" else token_down
        side_label = "UP"     if direction == "up" else "DOWN"
        cost_basis = shares * entry_price

        exit_price = self.simulate_fill_price(token_id, "SELL", cost_basis, get_price)
        if exit_price <= 0:
            exit_price = get_price(token_id, "BUY")
        if exit_price <= 0:
            exit_price = entry_price  # last resort: no loss/gain

        pnl = (exit_price - entry_price) * shares
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Close the most-recent open trade for this market+side
                conn.execute("""
                    UPDATE paper_trades
                    SET status='closed',
                        exit_price=?,
                        exit_timestamp=?,
                        pnl=?,
                        reason=?
                    WHERE id = (
                        SELECT id FROM paper_trades
                        WHERE market_id = ?
                          AND side      = ?
                          AND status    = 'open'
                        ORDER BY timestamp DESC
                        LIMIT 1
                    )
                """, (exit_price, now, pnl, reason, market_id, side_label))
                conn.commit()
        except sqlite3.Error as e:
            logger.debug("Error closing paper trade: %s", e)

        self.virtual_balance += exit_price * shares
        self._save_balance()

        sign = "+" if pnl >= 0 else ""
        msg = (f"[PAPER] ✓ CLOSE {side_label} | "
               f"{shares:.2f}sh @ ${exit_price:.4f} | P&L: {sign}${pnl:.2f}")
        return exit_price, pnl, msg

    def close_all(self, positions: list,
                  token_up: str, token_down: str,
                  market_id: str, get_price,
                  trade_logger, reason: str,
                  session_pnl: float,
                  trade_history: list) -> tuple[float, int, float, list]:
        """
        Close all open positions — mirrors close_all_positions() in trade_executor.

        Returns:
            (total_pnl, count, updated_session_pnl, pnl_list)
            pnl_list entries: (direction, shares, entry_price, exit_price, pnl)
        """
        total_pnl = 0.0
        count     = 0
        pnl_list  = []

        for p in list(positions):
            exit_price, pnl, _ = self.close_position(
                p['direction'], p['shares'], p['price'],
                token_up, token_down, market_id, get_price, reason,
            )
            total_pnl   += pnl
            session_pnl += pnl
            trade_history.append(pnl)
            pnl_list.append(
                (p['direction'], p['shares'], p['price'], exit_price, pnl)
            )
            trade_logger.log_trade(
                "CLOSE", p['direction'], p['shares'], exit_price,
                p['shares'] * exit_price,
                f"paper:{reason}", pnl, session_pnl,
            )
            count += 1

        positions.clear()
        return total_pnl, count, session_pnl, pnl_list

    # ── P&L / Portfolio ───────────────────────────────────────────────────────

    def calculate_unrealized_pnl(self, positions: list,
                                  token_up: str, token_down: str,
                                  get_price) -> float:
        """Compute total unrealized P&L across all open positions."""
        total = 0.0
        for p in positions:
            token_id      = token_up if p['direction'] == 'up' else token_down
            current_price = get_price(token_id, "SELL")
            if current_price > 0:
                total += (current_price - p['price']) * p['shares']
        return total

    def get_portfolio_summary(self) -> dict:
        """
        Return aggregate portfolio statistics from SQLite.

        Keys: total_trades, realized_pnl, wins, losses
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("""
                    SELECT
                        COUNT(*) AS total_trades,
                        COALESCE(SUM(CASE WHEN status = 'closed'
                                         THEN pnl ELSE 0 END), 0.0) AS realized_pnl,
                        COALESCE(SUM(CASE WHEN status = 'closed'
                                          AND pnl > 0
                                         THEN 1 ELSE 0 END), 0)    AS wins,
                        COALESCE(SUM(CASE WHEN status = 'closed'
                                          AND pnl <= 0
                                         THEN 1 ELSE 0 END), 0)    AS losses
                    FROM paper_trades
                """).fetchone()
            return {
                'total_trades': int(row[0]),
                'realized_pnl': float(row[1]),
                'wins':         int(row[2]),
                'losses':       int(row[3]),
            }
        except sqlite3.Error as e:
            logger.debug("Error fetching portfolio summary: %s", e)
            return {'total_trades': 0, 'realized_pnl': 0.0, 'wins': 0, 'losses': 0}
