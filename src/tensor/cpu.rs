//! Low-level tensor operations and representation on the CPU.

use std::{cmp, iter, num::NonZeroUsize, ops, sync::Arc};

use crate::tensor::{
    ops::{ToCpu, LL},
    typ::{Bool, Elem, Float, Num},
};

/// Low-level tensor operations on the CPU.
#[derive(Debug)]
pub struct TensorOps;

impl LL for TensorOps {
    type Repr<E> = Tensor<E>;

    fn new<E>(shape: &[NonZeroUsize], data: &[E]) -> Self::Repr<E>
    where
        E: Elem,
    {
        let layout = Layout::from(shape);
        assert_eq!(
            layout.capacity().get(),
            data.len(),
            "unexpected element count (want {}, but got {})",
            layout.capacity(),
            data.len()
        );
        Tensor {
            buffer: Arc::from(data.to_vec()),
            layout,
        }
    }

    fn convert<E, EInto>(t: &Self::Repr<E>) -> Self::Repr<EInto>
    where
        E: Elem + Into<EInto>,
        EInto: Elem,
    {
        t.map(|x| x.clone().into())
    }

    fn shape<E>(t: &Self::Repr<E>) -> &[NonZeroUsize] {
        &t.layout.shape
    }

    fn exp<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float,
    {
        t.map(|x| x.exp())
    }

    fn ln<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float,
    {
        t.map(|x| x.ln())
    }

    fn add<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Num,
    {
        lhs.zip(rhs, |x, y| ops::Add::add(x.clone(), y.clone()))
    }

    fn sub<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Num,
    {
        lhs.zip(rhs, |x, y| ops::Sub::sub(x.clone(), y.clone()))
    }

    fn mul<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Num,
    {
        lhs.zip(rhs, |x, y| ops::Mul::mul(x.clone(), y.clone()))
    }

    fn div<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Num,
    {
        lhs.zip(rhs, |x, y| ops::Div::div(x.clone(), y.clone()))
    }

    fn pow<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float,
    {
        lhs.zip(rhs, |&x, &y| num::Float::powf(x, y))
    }

    fn eq<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<bool>
    where
        E: Elem + PartialEq,
    {
        lhs.zip(rhs, |x, y| PartialEq::eq(x, y))
    }

    fn sum<E>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E>
    where
        E: Num,
    {
        t.reduce(
            axes,
            || E::zero(),
            |x, y| ops::Add::add(x.clone(), y.clone()),
        )
    }

    fn max<E>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E>
    where
        E: Bool,
    {
        t.reduce(
            axes,
            || E::min_value(),
            |x, y| {
                if PartialOrd::partial_cmp(x, y).is_none_or(cmp::Ordering::is_gt) {
                    x.clone()
                } else {
                    y.clone()
                }
            },
        )
    }

    fn reshape<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Self::Repr<E>
    where
        E: Elem,
    {
        match t.layout.reshape(shape) {
            Reshaped::Copy => Tensor {
                buffer: t.iter().cloned().collect(),
                layout: Layout::from(shape),
            },
            Reshaped::InPlace(layout) => Tensor {
                buffer: t.buffer.clone(),
                layout,
            },
        }
    }

    fn permute<E>(t: &Self::Repr<E>, permutation: &[usize]) -> Self::Repr<E> {
        let layout = t.layout.permute(permutation);
        Tensor {
            buffer: t.buffer.clone(),
            layout,
        }
    }

    fn expand<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Self::Repr<E> {
        let layout = t.layout.expand(shape);
        Tensor {
            buffer: t.buffer.clone(),
            layout,
        }
    }
}

impl ToCpu for TensorOps {
    fn to_cpu<E>(t: &Self::Repr<E>) -> self::Tensor<E> {
        t.clone()
    }
}

/// Low-level tensor representation on the CPU.
#[derive(Debug)]
pub struct Tensor<E> {
    buffer: Arc<[E]>,
    layout: Layout,
}

impl<E> Clone for Tensor<E> {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            layout: self.layout.clone(),
        }
    }
}

impl<'a, E> IntoIterator for &'a Tensor<E> {
    type Item = &'a E;

    type IntoIter = TensorIter<'a, E>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            tensor: self,
            indices: self.layout.iter(),
        }
    }
}

impl<E> Tensor<E> {
    /// Create row-major iterator over the tensor.
    #[must_use]
    pub fn iter(&self) -> TensorIter<'_, E> {
        self.into_iter()
    }

    /// Collect all elements of the tensor into a [`Vec`].
    #[must_use]
    pub fn ravel(&self) -> Vec<E>
    where
        E: Clone,
    {
        self.iter().cloned().collect()
    }

    fn map<F, T>(&self, op: F) -> Tensor<T>
    where
        F: Fn(&E) -> T,
    {
        let buffer = self.iter().map(op);
        Tensor {
            buffer: buffer.collect(),
            layout: Layout::from(self.layout.shape.clone()),
        }
    }

    fn zip<F, T>(&self, other: &Self, op: F) -> Tensor<T>
    where
        F: Fn(&E, &E) -> T,
    {
        assert_eq!(
            self.layout.shape, other.layout.shape,
            "zip: incompatible shapes ({:?} and {:?})",
            self.layout.shape, other.layout.shape
        );
        let buffer = self.iter().zip(other.iter()).map(|(x, y)| op(x, y));
        Tensor {
            buffer: buffer.collect(),
            layout: Layout::from(self.layout.shape.clone()),
        }
    }

    fn reduce<D, F, T>(&self, axes: &[usize], default: D, op: F) -> Tensor<T>
    where
        D: Fn() -> T,
        F: Fn(&T, &E) -> T,
    {
        let (layout, reducer) = self.layout.reduce(axes);
        let mut buffer: Vec<_> = iter::repeat_with(default)
            .take(layout.capacity().get())
            .collect();
        for idx in &self.layout {
            let dst_pos = reducer.translate(&idx);
            let src_pos = self.layout.translate(&idx);
            buffer[dst_pos] = op(&buffer[dst_pos], &self.buffer[src_pos]);
        }
        Tensor {
            buffer: buffer.into(),
            layout,
        }
    }
}

/// A row-major iterator over a tensor.
#[derive(Debug)]
pub struct TensorIter<'a, E> {
    tensor: &'a Tensor<E>,
    indices: IndexIter<'a>,
}

impl<'a, T> Iterator for TensorIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices
            .next()
            .map(|idx| &self.tensor.buffer[self.tensor.layout.translate(&idx)])
    }
}

/// A layout describes how a tensor is laid out on the CPU's memory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Layout {
    /// The number of elements in each axis.
    shape: Box<[NonZeroUsize]>,

    /// The number of elements in the memory array that need to be skipped to move to the next
    /// element in each axis.
    strides: Box<[usize]>,
}

impl<T> From<T> for Layout
where
    T: Into<Box<[NonZeroUsize]>>,
{
    fn from(shape: T) -> Self {
        let shape = shape.into();
        if shape.is_empty() {
            return Self::scalar();
        }
        Self::contiguous(shape)
    }
}

impl<'a> IntoIterator for &'a Layout {
    type Item = Box<[usize]>;
    type IntoIter = IndexIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IndexIter {
            layout: self,
            index: Box::from(vec![0; self.shape.len()]),
            exhausted: false,
        }
    }
}

impl Layout {
    /// Creates a contiguous row-major layout based on the given shape.
    fn contiguous(shape: Box<[NonZeroUsize]>) -> Self {
        assert!(!shape.is_empty(), "shape must not be empty");
        // Go backwards through the shape to calculate the strides. The last strides is always 1.
        let mut strides = vec![1; shape.len()].into_boxed_slice();
        for idx in (0..shape.len() - 1).rev() {
            strides[idx] = strides[idx + 1] * shape[idx + 1].get();
        }
        Self { shape, strides }
    }

    /// Returns the layout for a scalar, which has no shape nor strides.
    fn scalar() -> Self {
        Self::default()
    }

    /// Creates a row-major iterator over all indices of the tensor.
    fn iter(&self) -> IndexIter<'_> {
        self.into_iter()
    }

    /// Returns the number of elements in the tensor having this layout.
    fn capacity(&self) -> NonZeroUsize {
        self.shape
            .iter()
            .copied()
            .reduce(|x, y| x.checked_mul(y).expect("no overflow"))
            .unwrap_or(NonZeroUsize::MIN)
    }

    /// Translates a tensor index into a position in the data buffer.
    fn translate(&self, index: &[usize]) -> usize {
        let index_it = index.iter().rev();
        let strides_it = self.strides.iter().rev();
        index_it.zip(strides_it).map(|(x, s)| x * s).sum()
    }

    /// Returns 2 layouts where the first is reduced layout and the second is the reducer layout.
    /// The reducer layout is used to map an index in the original tensor to a memory position in
    /// the reduced tensor.
    #[allow(clippy::similar_names)]
    fn reduce(&self, axes: &[usize]) -> (Self, Self) {
        let mut reduced_shape = self.shape.clone();
        for &d in axes {
            reduced_shape[d] = NonZeroUsize::MIN;
        }
        let reduced_layout = Self::from(reduced_shape);
        let mut reducer_layout = reduced_layout.clone();
        for &d in axes {
            // The reducer layout is similar to the reduced layout, except that the strides of the
            // reduced axes are set to 0. This prevents that dimension from contributing to the data
            // position. Thus, we can map multiple elements along a dimension in the original tensor
            // to the same memory position in the reduced tensor.
            reducer_layout.strides[d] = 0;
        }
        (reduced_layout, reducer_layout)
    }

    /// Returns a new layout where all singleton axes are removed.
    fn squeeze(&self) -> Self {
        let mut shape = Vec::with_capacity(self.shape.len());
        let mut strides = Vec::with_capacity(self.strides.len());
        for (&size, &stride) in self.shape.iter().zip(self.strides.iter()) {
            if size > NonZeroUsize::MIN {
                shape.push(size);
                strides.push(stride);
            }
        }
        Self {
            shape: Box::from(shape),
            strides: Box::from(strides),
        }
    }

    /// Returns a new layout where the dimensions are permuted.
    ///
    /// # Errors
    ///
    /// Returns an error if one of the dimensions is invalid.
    fn permute(&self, permutation: &[usize]) -> Self {
        let rank = self.shape.len();
        let mut shape = Vec::with_capacity(rank);
        let mut strides = Vec::with_capacity(rank);
        for &axis in &permutation[..rank] {
            shape.push(self.shape[axis]);
            strides.push(self.strides[axis]);
        }
        Self {
            shape: Box::from(shape),
            strides: Box::from(strides),
        }
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
    fn expand(&self, new_shape: &[NonZeroUsize]) -> Self {
        let mut new_strides = vec![0; new_shape.len()];
        for dim in 0..self.shape.len() {
            let old_idx = self.shape.len() - dim - 1;
            let new_idx = new_shape.len() - dim - 1;
            if self.shape[old_idx] == new_shape[new_idx] {
                new_strides[new_idx] = self.strides[old_idx];
            } else if self.shape[old_idx] == NonZeroUsize::MIN {
                new_strides[new_idx] = 0;
            } else {
                panic!(
                    "expand: incompatible shapes ({:?} and {:?})",
                    self.shape, new_shape
                );
            }
        }
        Self {
            shape: Box::from(new_shape),
            strides: Box::from(new_strides),
        }
    }

    /// Returns a new layout for a tensor having the same number of elements
    /// but with a different shape. This function returns an error if the new
    /// layout can't be accommodated without copying data.
    ///
    /// # Errors
    ///
    /// Returns an error if the new shape is incompatible with the layout.
    fn reshape(&self, new_shape: &[NonZeroUsize]) -> Reshaped {
        let new_capacity = new_shape
            .iter()
            .copied()
            .reduce(|x, y| x.checked_mul(y).expect("no overflow"))
            .unwrap_or(NonZeroUsize::MIN);

        assert_eq!(
            new_capacity,
            self.capacity(),
            "reshape: incompatible shapes ({:?} and {:?})",
            self.shape,
            new_shape
        );

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
                    old_size = old_size
                        .checked_mul(old_layout.shape[old_dim])
                        .expect("no overflow");
                } else {
                    new_dim += 1;
                    new_size = new_size
                        .checked_mul(new_shape[new_dim])
                        .expect("no overflow");
                }
            }
            // Check if the reshaped dimensions are non-contiguous in memory.
            for dim in old_dim_prev..old_dim {
                let expected_stride = old_layout.strides[dim + 1] * old_layout.shape[dim + 1].get();
                if old_layout.strides[dim] != expected_stride {
                    return Reshaped::Copy;
                }
            }
            // Build a strides backward.
            new_strides[new_dim] = old_layout.strides[old_dim];
            for dim in (new_dim_prev..new_dim).rev() {
                new_strides[dim] = new_strides[dim + 1] * new_shape[dim + 1].get();
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
        Reshaped::InPlace(Self {
            shape: Box::from(new_shape),
            strides: Box::from(new_strides),
        })
    }
}

/// An iterator over a tensor's indices.
#[derive(Debug)]
struct IndexIter<'a> {
    layout: &'a Layout,
    index: Box<[usize]>,
    exhausted: bool,
}

impl Iterator for IndexIter<'_> {
    type Item = Box<[usize]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let index = self.index.clone();
        for (i, s) in self.layout.shape.iter().enumerate().rev() {
            self.index[i] += 1;
            if self.index[i] < s.get() {
                return Some(index);
            }
            self.index[i] = 0;
        }
        self.exhausted = true;
        Some(index)
    }
}

enum Reshaped {
    Copy,
    InPlace(Layout),
}
