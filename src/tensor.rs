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
            return Err(TensorError::IncompatibleShapes(
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
    pub fn broadcast<F>(&self, other: &Self, op: F) -> Result<Self, TensorError>
    where
        F: Fn(&f32, &f32) -> f32,
    {
        let broadcasted_shape = broadcast_shape(self.layout.shape(), other.layout.shape())?;
        let lhs = self.expand(&broadcasted_shape)?;
        let rhs = other.expand(&broadcasted_shape)?;
        Ok(lhs.zip(&rhs, op))
    }

    /// Apply the unary function `op` to all elements in the tensor.
    pub fn map(&self, op: impl Fn(&f32) -> f32) -> Self {
        let mut res = Vec::with_capacity(self.layout.elems());
        for x in self.into_iter() {
            res.push(op(x));
        }
        Self {
            data: Arc::new(res),
            layout: TensorLayout::from(self.layout.shape()),
        }
    }
    /// Apply the unary function `op` to all elements in the tensor.
    pub fn zip(&self, other: &Self, op: impl Fn(&f32, &f32) -> f32) -> Self {
        let mut res = Vec::with_capacity(self.layout.elems().min(other.layout.elems()));
        for (x, y) in self.into_iter().zip(other.into_iter()) {
            res.push(op(x, y));
        }
        Self {
            data: Arc::new(res),
            layout: TensorLayout::from(self.layout.shape()),
        }
    }

    /// Transposes the tensor without cloning its data.
    pub fn transpose(&self) -> Self {
        let layout = self.layout.transpose();
        Self {
            data: self.data.clone(),
            layout,
        }
    }

    /// Transposes the tensor without cloning its data.
    pub fn permute(&self, permutation: &[usize]) -> Result<Self, TensorError> {
        let layout = self.layout.permute(permutation)?;
        Ok(Self {
            data: self.data.clone(),
            layout,
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
