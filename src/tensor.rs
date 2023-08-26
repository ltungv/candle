//! An N-dimension tensor.

pub mod error;
pub mod layout;

use std::sync::Arc;

use self::{
    error::TensorError,
    layout::{broadcast_shape, PositionIterator, TensorLayout},
};

/// An N-dimension array.
#[derive(Debug)]
pub struct Tensor {
    data: Arc<Vec<f32>>,
    layout: TensorLayout,
}

impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Self {
        let data_len = data.len();
        Self {
            data: Arc::new(data),
            layout: TensorLayout::from(&[data_len]),
        }
    }
}

impl From<&[f32]> for Tensor {
    fn from(data: &[f32]) -> Self {
        Tensor::from(data.to_vec())
    }
}

impl<const N: usize> From<[f32; N]> for Tensor {
    fn from(data: [f32; N]) -> Self {
        Tensor::from(data.to_vec())
    }
}

impl<'a> IntoIterator for &'a Tensor {
    type Item = &'a f32;

    type IntoIter = TensorIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            tensor: self,
            position_iterator: self.layout.iter_position(),
        }
    }
}

impl Tensor {
    /// Creates a new tensor using the given data and layout.
    pub fn new(data: &[f32], layout: TensorLayout) -> Result<Self, TensorError> {
        if layout.elems() != data.len() {
            return Err(TensorError::ShapeMismatch(
                layout.shape().to_vec(),
                vec![data.len()],
            ));
        }
        Ok(Self {
            data: Arc::new(data.to_vec()),
            layout,
        })
    }

    /// Returns the layout of this tensor.
    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    /// Apply the binary function `op` to the two tensors, performing broadcast
    /// when neccessary.
    pub fn broadcasted_apply<F>(&self, other: &Self, op: F) -> Result<Self, TensorError>
    where
        F: Fn(&f32, &f32) -> f32,
    {
        let broadcasted_shape = broadcast_shape(self.layout.shape(), other.layout.shape())?;
        let mut res = Vec::with_capacity(broadcasted_shape.iter().product());
        let lhs = self.expand(&broadcasted_shape).unwrap();
        let rhs = other.expand(&broadcasted_shape).unwrap();
        for (x, y) in lhs.into_iter().zip(rhs.into_iter()) {
            res.push(op(x, y));
        }
        Ok(Self {
            data: Arc::new(res),
            layout: TensorLayout::from(broadcasted_shape),
        })
    }

    /// Expands the tensor to the given shape without cloning its data.
    fn expand(&self, shape: &[usize]) -> Result<Self, TensorError> {
        let layout = self.layout.expand(shape)?;
        Ok(Self {
            data: self.data.clone(),
            layout,
        })
    }
}

/// An iterator over a tensor.
pub struct TensorIterator<'a> {
    tensor: &'a Tensor,
    position_iterator: PositionIterator<'a>,
}

impl<'a> Iterator for TensorIterator<'a> {
    type Item = &'a f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.position_iterator
            .next()
            .map(|pos| &self.tensor.data[pos])
    }
}
