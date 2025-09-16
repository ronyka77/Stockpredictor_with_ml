CREATE TABLE dividends (
  id VARCHAR(128) PRIMARY KEY,            
  ticker_id INT NOT NULL,         
  cash_amount NUMERIC(18,6) NOT NULL,     
  currency CHAR(3),                      
  declaration_date DATE,
  ex_dividend_date DATE,
  pay_date DATE,
  record_date DATE,
  frequency SMALLINT,                    
  dividend_type VARCHAR(16),              
  raw_payload JSONB NOT NULL,            
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  CONSTRAINT fk_ticker FOREIGN KEY (ticker_id) REFERENCES tickers(id)
);

CREATE INDEX idx_dividends_ticker ON dividends(ticker_id);
CREATE INDEX idx_dividends_ex_date ON dividends(ex_dividend_date);
CREATE UNIQUE INDEX ux_dividends_ticker_ex_pay_amt
  ON dividends(ticker_id, ex_dividend_date, pay_date, cash_amount);