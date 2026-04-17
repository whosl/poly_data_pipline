use pyo3::prelude::*;

mod orderbook;
mod timestamp;
mod types;

#[pymodule]
fn poly_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(timestamp::now_ns, m)?)?;
    m.add_class::<orderbook::OrderBook>()?;
    Ok(())
}
