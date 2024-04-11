//! An N-dimension tensor.

mod error;
mod layout;

use std::sync::Arc;

pub use error::Error;
pub use layout::Layout;
use rand::Rng;
use rand_distr::Distribution;

use self::layout::Iter;

/// An N-dimension array holding elements row-major order. Tensors are immutable and new ones are
/// created each time we perform an operation. Tensors' underlying data is shared using reference
/// counting and only cloned when an operations can't be performed without modifying the data.
#[derive(Debug)]
pub struct Tensor {
    data: Arc<[f32]>,
    layout: Layout,
}

impl std::ops::Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Self::Output {
        self.safe_add(other).unwrap()
    }
}

impl std::ops::Add<Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Self::Output {
        self.safe_add(&other).unwrap()
    }
}

impl std::ops::Add for Tensor {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        self.safe_add(&other).unwrap()
    }
}

impl std::ops::Add<&Self> for Tensor {
    type Output = Self;

    fn add(self, other: &Self) -> Self::Output {
        self.safe_add(other).unwrap()
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        self.safe_mul(other).unwrap()
    }
}

impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        self.safe_mul(&other).unwrap()
    }
}

impl std::ops::Mul for Tensor {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        self.safe_mul(&other).unwrap()
    }
}

impl std::ops::Mul<&Self> for Tensor {
    type Output = Self;

    fn mul(self, other: &Self) -> Self::Output {
        self.safe_mul(other).unwrap()
    }
}

impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Self {
        let data_len = data.len();
        Self {
            data: Arc::from(data),
            layout: Layout::from(&[data_len]),
        }
    }
}

impl From<&[f32]> for Tensor {
    fn from(data: &[f32]) -> Self {
        Self::from(data.to_vec())
    }
}

impl<const N: usize> From<[f32; N]> for Tensor {
    fn from(data: [f32; N]) -> Self {
        Self::from(data.to_vec())
    }
}

impl<const N: usize> From<&[f32; N]> for Tensor {
    fn from(data: &[f32; N]) -> Self {
        Self::from(data.to_vec())
    }
}

impl std::ops::Index<usize> for &Tensor {
    type Output = f32;

    fn index(&self, pos: usize) -> &Self::Output {
        &self.data[pos]
    }
}

impl std::ops::Index<&[usize]> for &Tensor {
    type Output = f32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let pos = self.layout.translate(index);
        &self[pos]
    }
}

impl std::ops::Index<Vec<usize>> for &Tensor {
    type Output = f32;

    fn index(&self, index: Vec<usize>) -> &Self::Output {
        &self[index.as_slice()]
    }
}

impl<const N: usize> std::ops::Index<[usize; N]> for &Tensor {
    type Output = f32;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self[&index]
    }
}

impl<const N: usize> std::ops::Index<&[usize; N]> for &Tensor {
    type Output = f32;

    fn index(&self, index: &[usize; N]) -> &Self::Output {
        &self[index.as_slice()]
    }
}

impl<'a> IntoIterator for &'a Tensor {
    type Item = f32;

    type IntoIter = RowIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            tensor: self,
            index_iterator: self.layout.iter(),
        }
    }
}

impl Tensor {
    /// Creates a new tensor holding a scalar.
    #[must_use]
    pub fn scalar(x: f32) -> Self {
        Self {
            data: Arc::from(vec![x]),
            layout: Layout::scalar(),
        }
    }

    /// Creates a new tensor using the given data and layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout and data are incompatible.
    pub fn shaped(shape: &[usize], data: &[f32]) -> Result<Self, Error> {
        let layout = Layout::from(shape);
        if layout.elems() != data.len() {
            return Err(Error::IncompatibleShapes(shape.to_vec(), vec![data.len()]));
        }
        Ok(Self {
            data: Arc::from(data.to_vec()),
            layout,
        })
    }

    /// Creates a new tensor with randomized data.
    pub fn rand<R, D>(rng: R, distribution: D, shape: &[usize]) -> Self
    where
        R: Rng,
        D: Distribution<f32>,
    {
        let layout = Layout::from(shape);
        let data: Vec<_> = rng.sample_iter(distribution).take(layout.elems()).collect();
        Self {
            data: Arc::from(data),
            layout,
        }
    }

    /// Returns the layout of this tensor.
    #[must_use]
    pub const fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Matrix product of two arrays.
    ///
    /// The behavior depends on the arguments in the following ways:
    /// + If both arguments are 2-D they are multiplied like conventional matrices.
    /// + If either argument is N-D, N > 2, it is treated as a stack of matrices
    /// residing in the last two indexes and broadcast accordingly.
    /// + If the first argument is 1-D, it is promoted to a matrix by prepending
    /// a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
    /// + If the second argument is 1-D, it is promoted to a matrix by appending a 1
    /// to its dimensions. After matrix multiplication the appended 1 is removed.
    ///
    /// `matmul` differs from dot in two important ways:
    /// + Multiplication by scalars is not allowed, use * instead.
    /// + Stacks of matrices are broadcast together as if the matrices were elements,
    /// respecting the signature (n,k),(k,m)->(n,m)
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes of the tensors are incompatible.
    pub fn matmul(&self, other: &Self) -> Result<Self, Error> {
        let mut lhs_shape = self.layout.shape().to_vec();
        let mut rhs_shape = other.layout.shape().to_vec();
        let orig_lhs_dims = lhs_shape.len();
        let orig_rhs_dims = rhs_shape.len();
        if orig_lhs_dims == 0 || orig_rhs_dims == 0 {
            return Err(Error::IncompatibleShapes(lhs_shape, rhs_shape));
        }
        // If the LHS dimension is (k), make it (1, k)
        if orig_lhs_dims == 1 {
            lhs_shape.insert(0, 1);
        }
        // If the RHS dimension is (k), make it (k, 1)
        if orig_rhs_dims == 1 {
            rhs_shape.push(1);
        }
        if lhs_shape[lhs_shape.len() - 1] != rhs_shape[rhs_shape.len() - 2] {
            return Err(Error::IncompatibleShapes(lhs_shape, rhs_shape));
        }
        // Turn (..., m, k) into (..., m, 1, k);
        lhs_shape.insert(lhs_shape.len() - 1, 1);
        // Turn (..., k, n) into (..., 1, k, n);
        rhs_shape.insert(rhs_shape.len() - 2, 1);
        // Multiply (..., m, 1, k) with (..., 1, n, k) to get (..., m, n, k)
        let lhs = self.reshape(&lhs_shape)?;
        let rhs = other.reshape(&rhs_shape)?;
        let broadcasted_mul =
            &lhs.safe_mul(&rhs.transpose(rhs_shape.len() - 1, rhs_shape.len() - 2)?)?;
        // Sum the last dimension to get (..., m, n, 1)
        let sum_last_dim = broadcasted_mul.sum(&[broadcasted_mul.layout.shape().len() - 1])?;
        // Remove last dimension
        let mut shape = {
            let s = sum_last_dim.layout.shape();
            s[..s.len() - 1].to_vec()
        };
        // Remove prepended dimension if necessary
        if orig_lhs_dims == 1 {
            shape.remove(shape.len() - 2);
        }
        // Remove appended dimension if necessary
        if orig_rhs_dims == 1 {
            shape.remove(shape.len() - 1);
        }
        sum_last_dim.reshape(&shape)
    }

    /// Returns a new tensor resulted from adding the elements of `self` and `other`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensors cannot be broadcasted.
    pub fn safe_add(&self, other: &Self) -> Result<Self, Error> {
        self.zip(other, |x, y| x + y)
    }

    /// Returns a new tensor resulted from multiplying the elements of `self` and `other`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensors cannot be broadcasted.
    pub fn safe_mul(&self, other: &Self) -> Result<Self, Error> {
        self.zip(other, |x, y| x * y)
    }

    /// Returns a new tensor reduced along the given dimensions by summing all elements.
    ///
    /// # Errors
    ///
    /// Returns an error if one of the dimensions is invalid.
    pub fn sum(&self, dims: &[usize]) -> Result<Self, Error> {
        self.reduce(dims, 0.0, |x, y| x + y)
    }

    /// Returns a new tensor reduced along the given dimensions by multiplying all elements.
    ///
    /// # Errors
    ///
    /// Returns an error if one of the dimensions is invalid.
    pub fn prod(&self, dims: &[usize]) -> Result<Self, Error> {
        self.reduce(dims, 1.0, |x, y| x * y)
    }

    /// Applies the unary function `op` to all elements in the tensor.
    #[must_use]
    pub fn map<F>(&self, op: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let mut res = Vec::with_capacity(self.layout.elems());
        for x in self {
            res.push(op(x));
        }
        Self {
            data: Arc::from(res),
            layout: Layout::from(self.layout.shape()),
        }
    }

    /// Applies the binary function `op` by pairing each element in `self` and `other` and applying
    /// broadcast when necessary. See [NumPy's broadcasting] for more information.
    ///
    /// [NumPy's broadcasting]: https://numpy.org/doc/stable/user/basics.broadcasting.html
    ///
    /// # Errors
    ///
    /// Returns an error if the tensors cannot be broadcasted.
    pub fn zip<F>(&self, other: &Self, op: F) -> Result<Self, Error>
    where
        F: Fn(f32, f32) -> f32,
    {
        let (lhs, rhs) = self.broadcast(other)?;
        let mut res = Vec::with_capacity(lhs.layout.elems());
        for (x, y) in lhs.into_iter().zip(rhs.into_iter()) {
            res.push(op(x, y));
        }
        Ok(Self {
            data: Arc::from(res),
            layout: Layout::from(lhs.layout.shape()),
        })
    }

    /// Reduces all elements along the given dimensions into a single element using the given
    /// operation. This effectively reduces the rank of the tensor by the number of input
    /// dimensions. See [NumPy's reduce] for more information.
    ///
    /// [NumPy's reduce]: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduce.html#numpy-ufunc-reduce
    ///
    /// # Errors
    ///
    /// Returns an error if one of the dimensions is invalid.
    pub fn reduce<F>(&self, dims: &[usize], default: f32, op: F) -> Result<Self, Error>
    where
        F: Fn(&f32, &f32) -> f32,
    {
        let (layout, reducer) = self.layout.reduce(dims)?;
        let mut res = vec![default; layout.elems()];
        for idx in &self.layout {
            let src_pos = self.layout.translate(&idx);
            let dst_pos = reducer.translate(&idx);
            res[dst_pos] = op(&res[dst_pos], &self.data[src_pos]);
        }
        Ok(Self {
            data: Arc::from(res),
            layout,
        })
    }

    /// Removes all singleton dimensions from the tensor.
    #[must_use]
    pub fn squeeze(&self) -> Self {
        let layout = self.layout.squeeze();
        Self {
            data: self.data.clone(),
            layout,
        }
    }

    /// Swaps 2 dimensions of the tensor without cloning its data.
    ///
    /// # Errors
    ///
    /// Returns an error if one of the dimensions is invalid.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, Error> {
        let layout = self.layout.transpose(dim0, dim1)?;
        Ok(Self {
            data: self.data.clone(),
            layout,
        })
    }

    /// Permutes the tensor dimensions according to the given ordering without cloning its data.
    ///
    /// # Errors
    ///
    /// Returns an error if the permutation contains an invalid dimension.
    pub fn permute(&self, permutation: &[usize]) -> Result<Self, Error> {
        let layout = self.layout.permute(permutation)?;
        Ok(Self {
            data: self.data.clone(),
            layout,
        })
    }

    /// Reshapes the tensor to the given shape. This might clone the data if the new shape can't be
    /// represented contiguously basing on the current layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the new shape is incompatible with the current layout.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, Error> {
        self.layout.reshape(shape)?.map_or_else(
            || Self::from(self.data.as_ref()).reshape(shape),
            |layout| {
                Ok(Self {
                    data: self.data.clone(),
                    layout,
                })
            },
        )
    }

    /// Returns a row-wise iterator over all elements in the tensor.
    #[must_use]
    pub fn iter(&self) -> RowIter<'_> {
        self.into_iter()
    }

    /// Broadcast the tensors and returns their broadcasted versions. See [`TensorLayout::broadcast`]
    /// for more details.
    fn broadcast(&self, other: &Self) -> Result<(Self, Self), Error> {
        let (lhs_layout, rhs_layout) = self.layout.broadcast(&other.layout)?;
        let lhs = Self {
            data: self.data.clone(),
            layout: lhs_layout,
        };
        let rhs = Self {
            data: other.data.clone(),
            layout: rhs_layout,
        };
        Ok((lhs, rhs))
    }
}

/// A row-major iterator over a tensor.
#[derive(Debug)]
pub struct RowIter<'a> {
    tensor: &'a Tensor,
    index_iterator: Iter<'a>,
}

impl<'a> Iterator for RowIter<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iterator.next().map(|idx| self.tensor[idx])
    }
}
