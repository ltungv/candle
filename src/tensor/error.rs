//! Defines tensor errors.

use std::{error, fmt};

/// An error for operations on tensor(s).
#[derive(Clone, Debug)]
pub enum TensorError {
    /// An operation was performed on 2 objects with incompatible shapes.
    IncompatibleShapes(Vec<usize>, Vec<usize>),
    /// Argument to a function is invalid.
    InvalidArgument(String),
}

impl error::Error for TensorError {}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleShapes(l, r) => {
                write!(f, "Incompatible shapes: {:?} and {:?}.", l, r)
            }
            Self::InvalidArgument(s) => write!(f, "Invalid argument: {:?}.", s),
        }
    }
}
