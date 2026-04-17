use rust_decimal::Decimal;
use std::cmp::Ordering;

/// Wrapper for Decimal that reverses ordering (for bids: highest price first).
#[derive(Clone, Debug)]
pub struct ReverseDecimal(pub Decimal);

impl PartialEq for ReverseDecimal {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for ReverseDecimal {}

impl PartialOrd for ReverseDecimal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReverseDecimal {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0) // reversed: higher price comes first
    }
}
