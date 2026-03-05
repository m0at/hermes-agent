use pyo3::prelude::*;

mod prompt_scanner;
mod token_estimate;

#[pymodule]
fn hermes_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prompt_scanner::scan_context_content, m)?)?;
    m.add_function(wrap_pyfunction!(prompt_scanner::truncate_content, m)?)?;
    m.add_function(wrap_pyfunction!(token_estimate::estimate_tokens_rough, m)?)?;
    m.add_function(wrap_pyfunction!(token_estimate::estimate_messages_tokens_rough, m)?)?;
    Ok(())
}
