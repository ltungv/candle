//! Describes how N-dimension arrays are laid out in memory.

use super::error::Error;

/// A layout describes how an N-dimensional array is laid out in memory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    /// The number of elements in each dimension.
    shape: Box<[usize]>,
    /// The number of elements in the memory array that need to be skipped to move to the next
    /// element in each dimension.
    strides: Box<[usize]>,
}

impl From<Box<[usize]>> for Layout {
    /// Creates a contiguous row-major layout based on the given shape.
    fn from(shape: Box<[usize]>) -> Self {
        if shape.is_empty() {
            return Self::default();
        }
        // Go backwards through the shape to calculate the strides. The last strides is always 1.
        let mut strides = Box::<[usize]>::from(vec![1; shape.len()]);
        for idx in (0..shape.len() - 1).rev() {
            strides[idx] = strides[idx + 1] * shape[idx + 1];
        }
        Self { shape, strides }
    }
}

impl From<&[usize]> for Layout {
    /// Creates a contiguous row-major layout based on the given shape.
    fn from(shape: &[usize]) -> Self {
        Self::from(Box::from(shape))
    }
}

impl From<Vec<usize>> for Layout {
    /// Creates a contiguous row-major layout based on the given shape.
    fn from(shape: Vec<usize>) -> Self {
        Self::from(Box::from(shape))
    }
}

impl<const N: usize> From<[usize; N]> for Layout {
    /// Creates a contiguous row-major layout based on the given shape.
    fn from(shape: [usize; N]) -> Self {
        Self::from(Box::from(shape.as_slice()))
    }
}

impl<const N: usize> From<&[usize; N]> for Layout {
    /// Creates a contiguous row-major layout based on the given shape.
    fn from(shape: &[usize; N]) -> Self {
        Self::from(Box::from(shape.as_slice()))
    }
}

impl<'a> IntoIterator for &'a Layout {
    type Item = Vec<usize>;
    type IntoIter = Iter<'a>;

    /// Creates a row-major iterator over all indices of the tensor.
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            layout: self,
            index: vec![0; self.shape.len()],
            exhausted: false,
        }
    }
}

impl Layout {
    /// Returns the layout for a scalar, which has no shape nor strides.
    #[must_use]
    pub fn scalar() -> Self {
        Self::default()
    }

    /// Returns the shape of the layout.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.shape.as_ref()
    }

    /// Returns the strides of the layout.
    #[must_use]
    pub fn strides(&self) -> &[usize] {
        self.strides.as_ref()
    }

    /// Returns the number of elements in the tensor having this layout.
    #[must_use]
    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns 2 layouts where the first is reduced layout and the second is the reducer layout.
    /// The reducer layout is used to map an index in the original tensor to a memory position in
    /// the reduced tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if one of the dimensions to reduce is invalid.
    pub fn reduce(&self, dims: &[usize]) -> Result<(Self, Self), Error> {
        let mut reduced_shape = self.shape.clone();
        for &d in dims {
            if d >= reduced_shape.len() {
                return Err(Error::UnknownDimension(d));
            }
            reduced_shape[d] = 1;
        }
        let reduced_layout = Self::from(reduced_shape);
        let mut reducer_layout = reduced_layout.clone();
        for &d in dims {
            // The reducer layout is similar to the reduced layout, except that the strides of the reduced
            // dimensions are set to 0. This prevents that dimension from contributing to the data position.
            // Thus, we can map multiple elements along a dimension in the original tensor to the same
            // memory position in the reduced tensor.
            reducer_layout.strides[d] = 0;
        }
        Ok((reduced_layout, reducer_layout))
    }

    /// Returns a new layout where all singleton dimensions are removed.
    #[must_use]
    pub fn squeeze(&self) -> Self {
        let mut shape = Vec::with_capacity(self.shape.len());
        let mut strides = Vec::with_capacity(self.strides.len());
        for (&dim_size, &dim_stride) in self.shape.iter().zip(self.strides.iter()) {
            if dim_size != 1 {
                shape.push(dim_size);
                strides.push(dim_stride);
            }
        }
        Self {
            shape: Box::from(shape),
            strides: Box::from(strides),
        }
    }

    /// Returns a new layout where the 2 dimensions are transposed.
    ///
    /// # Errors
    ///
    /// Returns an error if one of the dimensions is invalid.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, Error> {
        if dim0 >= self.shape.len() {
            return Err(Error::UnknownDimension(dim0));
        }
        if dim1 >= self.shape.len() {
            return Err(Error::UnknownDimension(dim1));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Ok(Self { shape, strides })
    }

    /// Returns a new layout where the dimensions are permuted.
    ///
    /// # Errors
    ///
    /// Returns an error if one of the dimensions is invalid.
    pub fn permute(&self, permutation: &[usize]) -> Result<Self, Error> {
        let mut shape = Vec::with_capacity(self.shape.len());
        let mut strides = Vec::with_capacity(self.strides.len());
        let mut sum_dim = 0;
        for &d in permutation {
            if d >= self.shape.len() {
                return Err(Error::UnknownDimension(d));
            }
            shape.push(self.shape[d]);
            strides.push(self.strides[d]);
            sum_dim += d;
        }
        if permutation.len() * (permutation.len() - 1) != 2 * sum_dim {
            return Err(Error::Custom(
                "Each dimension must be specified exactly once.".to_string(),
            ));
        }
        Ok(Self {
            shape: Box::from(shape),
            strides: Box::from(strides),
        })
    }

    /// Returns a new layout for a tensor with singleton dimensions expanded to a larger size.
    ///
    /// Tensor can also be expanded to a larger number of dimensions, and the new ones will be
    /// appended at the front. For the new dimensions, the size cannot be set to -1.
    ///
    /// Expanding a tensor does not allocate new memory, but only creates a new view on the
    /// existing tensor where a dimension of size one is expanded to a larger size by setting
    /// the strides to 0. Any dimension of size 1 can be expanded to an arbitrary value without
    /// allocating new memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout cannot be expanded to the new shape.
    pub fn expand(&self, new_shape: &[usize]) -> Result<Self, Error> {
        if new_shape.len() < self.shape.len() {
            return Err(Error::IncompatibleShapes(
                self.shape.to_vec(),
                new_shape.to_vec(),
            ));
        }
        let mut new_strides = vec![0; new_shape.len()];
        for dim in 0..self.shape.len() {
            let old_idx = self.shape.len() - dim - 1;
            let new_idx = new_shape.len() - dim - 1;
            if self.shape[old_idx] == new_shape[new_idx] {
                new_strides[new_idx] = self.strides[old_idx];
            } else if self.shape[old_idx] == 1 {
                new_strides[new_idx] = 0;
            } else {
                return Err(Error::IncompatibleShapes(
                    self.shape.to_vec(),
                    new_shape.to_vec(),
                ));
            }
        }
        Ok(Self {
            shape: Box::from(new_shape),
            strides: Box::from(new_strides),
        })
    }

    /// Performs broadcasting on the 2 layouts and returns their broadcasted versions.
    /// See [broadcasting rule] for more details.
    ///
    /// [broadcasting rule]: https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes are incompatible for broadcasting.
    pub fn broadcast(&self, other: &Self) -> Result<(Self, Self), Error> {
        // Determine which shape has more dimensions.
        let (small, large) = if self.shape.len() < other.shape.len() {
            (self.shape(), other.shape())
        } else {
            (other.shape(), self.shape())
        };
        // Zipping the 2 shapes in reverse order while filling in 1 for the missing dimensions.
        let mut broadcasted_shape = large.to_vec();
        for dim in 0..small.len() {
            let sm_idx = small.len() - dim - 1;
            let lg_idx = large.len() - dim - 1;
            let sm_size = small[sm_idx];
            let lg_size = large[lg_idx];
            if sm_size == 1 {
                broadcasted_shape[lg_idx] = lg_size;
            } else if lg_size == 1 || lg_size == sm_size {
                broadcasted_shape[lg_idx] = sm_size;
            } else {
                return Err(Error::IncompatibleShapes(small.to_vec(), large.to_vec()));
            }
        }
        let lhs = self.expand(broadcasted_shape.as_slice())?;
        let rhs = other.expand(broadcasted_shape.as_slice())?;
        Ok((lhs, rhs))
    }

    /// Returns a new layout for a tensor having the same number of elements
    /// but with a different shape. This function returns an error if the new
    /// layout can't be accommodated without copying data.
    ///
    /// # Errors
    ///
    /// Returns an error if the new shape is incompatible with the layout.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Option<Self>, Error> {
        if self.elems() != new_shape.iter().product() {
            return Err(Error::IncompatibleShapes(
                self.shape.to_vec(),
                new_shape.to_vec(),
            ));
        }
        let old_layout = self.squeeze();
        let mut new_strides = vec![1; new_shape.len()];
        let mut old_dim = 0;
        let mut new_dim = 0;
        while old_dim < old_layout.shape.len() && new_dim < new_shape.len() {
            // Find the combination of dimensions from the old and new shapes that have the same
            // number of elements.
            let old_dim_prev = old_dim;
            let new_dim_prev = new_dim;
            let mut old_size = old_layout.shape[old_dim];
            let mut new_size = new_shape[new_dim];
            while old_size != new_size {
                if old_size < new_size {
                    old_dim += 1;
                    old_size *= old_layout.shape[old_dim];
                } else {
                    new_dim += 1;
                    new_size *= new_shape[new_dim];
                }
            }
            // Check if the reshaped dimensions are non-contiguous in memory.
            for dim in old_dim_prev..old_dim {
                let expected_stride = old_layout.strides[dim + 1] * old_layout.shape[dim + 1];
                if old_layout.strides[dim] != expected_stride {
                    return Ok(None);
                }
            }
            // Build a strides backward.
            new_strides[new_dim] = old_layout.strides[old_dim];
            for dim in (new_dim_prev..new_dim).rev() {
                new_strides[dim] = new_strides[dim + 1] * new_shape[dim + 1];
            }
            old_dim += 1;
            new_dim += 1;
        }
        if new_dim > 0 {
            // Fill in the remaining strides.
            let last_stride = new_strides[new_dim - 1];
            for stride in new_strides.iter_mut().skip(new_dim) {
                *stride = last_stride;
            }
        }
        Ok(Some(Self {
            shape: Box::from(new_shape),
            strides: Box::from(new_strides),
        }))
    }

    /// Translates a tensor index into a position in the data buffer.
    #[must_use]
    pub fn translate(&self, index: &[usize]) -> usize {
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(x, s)| x * s)
            .sum()
    }

    /// Returns an iterator over the indices of a tensor.
    #[must_use]
    pub fn iter(&self) -> Iter<'_> {
        self.into_iter()
    }
}

/// An iterator over a tensor's indices.
#[derive(Debug)]
pub struct Iter<'a> {
    layout: &'a Layout,
    index: Vec<usize>,
    exhausted: bool,
}

impl<'a> Iterator for Iter<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let index = self.index.clone();
        for (i, s) in self.layout.shape.iter().enumerate().rev() {
            self.index[i] += 1;
            if self.index[i] < *s {
                break;
            }
            self.index[i] = 0;
        }
        self.exhausted = self.index.iter().all(|e| *e == 0);
        Some(index)
    }
}
