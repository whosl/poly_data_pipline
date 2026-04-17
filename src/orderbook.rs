use pyo3::prelude::*;
use rust_decimal::Decimal;
use std::collections::BTreeMap;

use crate::types::ReverseDecimal;

/// Parse a Decimal from string, returning None on failure.
fn parse_decimal(s: &str) -> Option<Decimal> {
    Decimal::from_str_exact(s).ok()
}

#[pyclass]
pub struct OrderBook {
    asset_id: String,
    /// Bids: price -> size. Uses ReverseDecimal so iter() yields best bid first.
    bids: BTreeMap<ReverseDecimal, Decimal>,
    /// Asks: price -> size. Standard ordering, lowest ask first.
    asks: BTreeMap<Decimal, Decimal>,
    last_update_ns: u64,
    last_exchange_ts: u64,
}

#[pymethods]
impl OrderBook {
    #[new]
    pub fn py_new(asset_id: String) -> Self {
        OrderBook {
            asset_id,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_ns: 0,
            last_exchange_ts: 0,
        }
    }

    /// Replace the entire orderbook with a snapshot.
    /// bids_data: Vec of (price_str, size_str)
    /// asks_data: Vec of (price_str, size_str)
    pub fn apply_snapshot(
        &mut self,
        bids_data: Vec<(String, String)>,
        asks_data: Vec<(String, String)>,
        exchange_ts: u64,
    ) {
        self.bids.clear();
        self.asks.clear();
        for (price_str, size_str) in bids_data {
            if let (Some(p), Some(s)) = (parse_decimal(&price_str), parse_decimal(&size_str)) {
                if s > Decimal::ZERO {
                    self.bids.insert(ReverseDecimal(p), s);
                }
            }
        }
        for (price_str, size_str) in asks_data {
            if let (Some(p), Some(s)) = (parse_decimal(&price_str), parse_decimal(&size_str)) {
                if s > Decimal::ZERO {
                    self.asks.insert(p, s);
                }
            }
        }
        self.last_exchange_ts = exchange_ts;
    }

    /// Apply a single price level delta.
    /// side: "buy" or "sell" (case insensitive)
    /// If size is "0", remove the level.
    pub fn apply_delta(
        &mut self,
        side: &str,
        price_str: String,
        size_str: String,
        exchange_ts: u64,
    ) {
        let price = match parse_decimal(&price_str) {
            Some(p) => p,
            None => return,
        };
        let size = match parse_decimal(&size_str) {
            Some(s) => s,
            None => return,
        };

        match side.to_lowercase().as_str() {
            "buy" => {
                if size == Decimal::ZERO {
                    self.bids.remove(&ReverseDecimal(price));
                } else {
                    self.bids.insert(ReverseDecimal(price), size);
                }
            }
            "sell" => {
                if size == Decimal::ZERO {
                    self.asks.remove(&price);
                } else {
                    self.asks.insert(price, size);
                }
            }
            _ => {}
        }
        self.last_exchange_ts = exchange_ts;
    }

    /// Returns (price_str, size_str) of best bid, or None.
    pub fn best_bid(&self) -> Option<(String, String)> {
        self.bids
            .iter()
            .next()
            .map(|(k, v)| (k.0.to_string(), v.to_string()))
    }

    /// Returns (price_str, size_str) of best ask, or None.
    pub fn best_ask(&self) -> Option<(String, String)> {
        self.asks
            .iter()
            .next()
            .map(|(k, v)| (k.to_string(), v.to_string()))
    }

    /// Returns midpoint as string, or None.
    pub fn midpoint(&self) -> Option<String> {
        let bb = self.bids.iter().next()?;
        let ba = self.asks.iter().next()?;
        let mid = (bb.0 .0 + ba.0) / Decimal::from(2);
        Some(mid.to_string())
    }

    /// Returns spread as string, or None.
    pub fn spread(&self) -> Option<String> {
        let bb = self.bids.iter().next()?;
        let ba = self.asks.iter().next()?;
        let spread = ba.0 - bb.0 .0;
        Some(spread.to_string())
    }

    /// Returns microprice as string, or None.
    /// microprice = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
    pub fn microprice(&self) -> Option<String> {
        let bb = self.bids.iter().next()?;
        let ba = self.asks.iter().next()?;
        let bid_price = bb.0 .0;
        let bid_size = *bb.1;
        let ask_price = ba.0;
        let ask_size = *ba.1;
        let total_size = bid_size + ask_size;
        if total_size == Decimal::ZERO {
            return None;
        }
        let mp = (bid_price * ask_size + ask_price * bid_size) / total_size;
        Some(mp.to_string())
    }

    /// Returns top-of-book imbalance as string, or None.
    /// imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    pub fn imbalance(&self) -> Option<String> {
        let bb = self.bids.iter().next()?;
        let ba = self.asks.iter().next()?;
        let bid_size = *bb.1;
        let ask_size = *ba.1;
        let total = bid_size + ask_size;
        if total == Decimal::ZERO {
            return None;
        }
        let imb = (bid_size - ask_size) / total;
        Some(imb.to_string())
    }

    /// Returns top N bid and ask levels as (price_str, size_str) tuples.
    pub fn top_n_levels(
        &self,
        n: usize,
    ) -> (Vec<(String, String)>, Vec<(String, String)>) {
        let bids: Vec<(String, String)> = self
            .bids
            .iter()
            .take(n)
            .map(|(k, v)| (k.0.to_string(), v.to_string()))
            .collect();
        let asks: Vec<(String, String)> = self
            .asks
            .iter()
            .take(n)
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        (bids, asks)
    }

    /// Returns a Python dict with all computed features.
    pub fn depth_summary(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut dict = pyo3::types::PyDict::new(py);

            let bb = self.bids.iter().next();
            let ba = self.asks.iter().next();

            match (bb, ba) {
                (Some((bp, bs)), Some((ap, as_))) => {
                    dict.set_item("best_bid", bp.0.to_string())?;
                    dict.set_item("best_bid_size", bs.to_string())?;
                    dict.set_item("best_ask", ap.to_string())?;
                    dict.set_item("best_ask_size", as_.to_string())?;

                    let spread = *ap - bp.0;
                    dict.set_item("spread", spread.to_string())?;

                    let mid = (bp.0 + *ap) / Decimal::from(2);
                    dict.set_item("midpoint", mid.to_string())?;

                    let total = *bs + *as_;
                    if total > Decimal::ZERO {
                        let mp = (bp.0 * *as_ + *ap * *bs) / total;
                        dict.set_item("microprice", mp.to_string())?;
                        let imb = (*bs - *as_) / total;
                        dict.set_item("imbalance", imb.to_string())?;
                    } else {
                        dict.set_item("microprice", py.None())?;
                        dict.set_item("imbalance", py.None())?;
                    }
                }
                _ => {
                    dict.set_item("best_bid", py.None())?;
                    dict.set_item("best_bid_size", py.None())?;
                    dict.set_item("best_ask", py.None())?;
                    dict.set_item("best_ask_size", py.None())?;
                    dict.set_item("spread", py.None())?;
                    dict.set_item("midpoint", py.None())?;
                    dict.set_item("microprice", py.None())?;
                    dict.set_item("imbalance", py.None())?;
                }
            }

            dict.set_item("total_bid_levels", self.bids.len())?;
            dict.set_item("total_ask_levels", self.asks.len())?;
            dict.set_item("last_exchange_ts", self.last_exchange_ts)?;
            dict.set_item("asset_id", self.asset_id.as_str())?;

            Ok(dict.into())
        })
    }

    /// Compute imbalance for top N levels on each side.
    pub fn top_n_imbalance(&self, n: usize) -> Option<String> {
        let bid_size: Decimal = self.bids.iter().take(n).map(|(_, s)| *s).sum();
        let ask_size: Decimal = self.asks.iter().take(n).map(|(_, s)| *s).sum();
        let total = bid_size + ask_size;
        if total == Decimal::ZERO {
            return None;
        }
        let imb = (bid_size - ask_size) / total;
        Some(imb.to_string())
    }

    #[getter]
    pub fn asset_id(&self) -> &str {
        &self.asset_id
    }

    #[getter]
    pub fn last_exchange_ts(&self) -> u64 {
        self.last_exchange_ts
    }
}
