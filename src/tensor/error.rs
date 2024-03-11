//! Defines tensor errors.

use std::{error, fmt};

/// An error type for all operations on tensors.
#[derive(Clone, Debug)]
pub enum Error {
    /// An operation was performed on 2 objects with incompatible shapes.
    IncompatibleShapes(Vec<usize>, Vec<usize>),
    /// An operation was performed with an dimension that does not exist.
    UnknownDimension(usize),
    /// A custom error message.
    Custom(String),
}

impl error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleShapes(l, r) => write!(f, "Incompatible shapes {:?} and {:?}.", l, r),
            Self::UnknownDimension(d) => write!(f, "Unknown dimension {}.", d),
            Self::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl From<String> for Error {
    fn from(err: String) -> Self {
        Self::Custom(err)
    }
}
