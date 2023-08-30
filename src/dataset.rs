//! Data structures and methods for dealing with datasets.

/// A sample input and output data.
#[derive(Debug, Clone)]
pub struct Sample<const M: usize, const N: usize> {
    /// The input data.
    pub input: [f64; M],
    /// The output data.
    pub output: [f64; N],
}
