use std::iter;

use super::error::TensorError;

#[derive(Debug, PartialEq)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Layout {
    pub fn new(shape: &[usize]) -> Self {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1;
        for s in shape.iter().rev() {
            strides.push(stride);
            stride *= s;
        }
        strides.reverse();
        Self {
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    pub fn strides(&self) -> &[usize] {
        self.strides.as_slice()
    }

    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn expand(&self, shape: &[usize]) -> Result<Self, TensorError> {
        let (lhs, rhs) = if self.shape.len() < shape.len() {
            (self.shape.as_slice(), shape)
        } else {
            (shape, self.shape.as_slice())
        };
        let mut new_shape = Vec::with_capacity(rhs.len());
        let mut new_strides = Vec::with_capacity(rhs.len());
        let mut stride = 1;
        for dims in lhs
            .iter()
            .rev()
            .chain(iter::once(&1usize).cycle())
            .zip(rhs.iter().rev())
        {
            match dims {
                (1, d) => {
                    new_shape.push(*d);
                    new_strides.push(stride);
                    stride *= *d;
                }
                (d, 1) => {
                    new_shape.push(*d);
                    new_strides.push(stride);
                    stride *= *d;
                }
                (dx, dy) if dx == dy => {
                    new_shape.push(*dx);
                    new_strides.push(stride);
                    stride *= *dx;
                }
                _ => return Err(TensorError::ShapeMismatch(lhs.to_vec(), rhs.to_vec())),
            }
        }
        new_shape.reverse();
        new_strides.reverse();
        Ok(Self {
            shape: new_shape,
            strides: new_strides,
        })
    }

    pub fn index_to_position(&self, index: &[usize]) -> usize {
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(x, s)| x * s)
            .sum()
    }

    pub fn position_to_index(&self, position: usize) -> Vec<usize> {
        let mut index = Vec::with_capacity(self.shape.len());
        let mut remainder = position;
        for s in self.strides() {
            index.push(remainder / s);
            remainder %= s;
        }
        index
    }
}
