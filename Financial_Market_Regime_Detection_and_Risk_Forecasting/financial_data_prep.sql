-- ==========================================
-- PART A: Daily End-of-Day (EOD) Architecture
-- ==========================================
CREATE TABLE asset_prices (
    date DATE,
    ticker VARCHAR(10),
    adj_close_price NUMERIC(12, 4),
    PRIMARY KEY (date, ticker)
);

CREATE OR REPLACE VIEW asset_returns AS
WITH PriceLag AS (
    SELECT 
        date,
        ticker,
        adj_close_price,
        LAG(adj_close_price) OVER (PARTITION BY ticker ORDER BY date) as prev_price
    FROM asset_prices
)
SELECT 
    date,
    ticker,
    adj_close_price,
    prev_price,
    LN(adj_close_price / prev_price) as log_return
FROM PriceLag
WHERE prev_price IS NOT NULL;

CREATE OR REPLACE VIEW annual_financial_metrics AS
SELECT 
    ticker,
    EXTRACT(YEAR FROM date) as trading_year,
    COUNT(log_return) as trading_days,
    SUM(log_return) as annual_log_return,
    AVG(log_return) as daily_mean_return,
    STDDEV_SAMP(log_return) as daily_volatility,
    STDDEV_SAMP(log_return) * SQRT(252) as annualized_volatility
FROM asset_returns
GROUP BY ticker, EXTRACT(YEAR FROM date);

-- ==========================================
-- PART B: Intraday High-Frequency Extension
-- ==========================================
CREATE TABLE intraday_ticks (
    tick_timestamp TIMESTAMP WITH TIME ZONE,
    ticker VARCHAR(10),
    price NUMERIC(12, 4),
    volume BIGINT,
    PRIMARY KEY (ticker, tick_timestamp)
);

-- Composite index for fast time-series slicing
CREATE INDEX idx_intraday_ticker_time ON intraday_ticks (ticker, tick_timestamp DESC);

-- Resample raw ticks into 1-minute OHLCV bars and compute intraday log returns
CREATE OR REPLACE VIEW intraday_1m_returns AS
WITH MinuteBars AS (
    SELECT 
        ticker,
        DATE_TRUNC('minute', tick_timestamp) AS bar_time,
        -- First and last price within each minute bucket
        (ARRAY_AGG(price ORDER BY tick_timestamp ASC))[1] AS open_price,
        MAX(price) AS high_price,
        MIN(price) AS low_price,
        (ARRAY_AGG(price ORDER BY tick_timestamp DESC))[1] AS close_price,
        SUM(volume) AS total_volume
    FROM intraday_ticks
    GROUP BY ticker, DATE_TRUNC('minute', tick_timestamp)
),
LaggedBars AS (
    SELECT 
        ticker,
        bar_time,
        close_price,
        total_volume,
        LAG(close_price) OVER (PARTITION BY ticker ORDER BY bar_time) AS prev_close
    FROM MinuteBars
)
SELECT 
    ticker,
    bar_time,
    close_price,
    total_volume,
    LN(close_price / prev_close) AS minute_log_return
FROM LaggedBars
WHERE prev_close IS NOT NULL AND prev_close > 0;

-- Compute Daily Realized Volatility from 1-Minute Intraday Returns
CREATE OR REPLACE VIEW daily_realized_volatility AS
SELECT 
    ticker,
    DATE(bar_time) AS trading_date,
    COUNT(minute_log_return) AS active_minutes,
    -- Realized variance is the sum of squared intraday log returns
    SQRT(SUM(POWER(minute_log_return, 2))) AS daily_realized_vol,
    SQRT(SUM(POWER(minute_log_return, 2)) * 252) AS annualized_realized_vol
FROM intraday_1m_returns
GROUP BY ticker, DATE(bar_time);