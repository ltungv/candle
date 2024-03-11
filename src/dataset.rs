//! Data structures and methods for dealing with datasets.

/// A single sample within a dataset of mappings from vectors to vectors.
#[derive(Debug, Clone)]
pub struct VectorMapping<const M: usize, const N: usize> {
    /// The input data.
    pub input: [f64; M],
    /// The output data.
    pub output: [f64; N],
}
