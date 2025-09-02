//! An N-dimensional array.

pub mod cpu;
pub mod ops;
pub mod typ;

use std::{marker::PhantomData, num::NonZeroUsize};

use ops::{ToCpu, ML};

use crate::tensor::typ::{Elem, Float, Num};

/// An alias for a tensor on the CPU having elements of type `E`.
pub type Cpu<E> = Tensor<cpu::Tensor<E>, E, cpu::TensorOps>;

/// A high-level tensor representation.
///
/// All methods on this struct are delegated to those in [`ML`], enabling auto-differentiation and
/// support for multiple hardwares. For convinience, operations are broadcasted by default, and
/// traits from [`std`] are implemented to overloads common operations for a numeric/container type.
#[derive(Debug)]
pub struct Tensor<T, E, Ops> {
    raw: T,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> std::ops::Add<Self> for &Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn add(self, other: Self) -> Self::Output {
        self.broadcast(other, Ops::add::<E>)
    }
}

impl<T, E, Ops> std::ops::Sub<Self> for &Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn sub(self, other: Self) -> Self::Output {
        self.broadcast(other, Ops::sub::<E>)
    }
}

impl<T, E, Ops> std::ops::Mul<Self> for &Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn mul(self, other: Self) -> Self::Output {
        self.broadcast(other, Ops::mul::<E>)
    }
}

impl<T, E, Ops> std::ops::Div<Self> for &Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn div(self, other: Self) -> Self::Output {
        self.broadcast(other, Ops::div::<E>)
    }
}

impl<T, E, Ops> Tensor<T, E, Ops> {
    /// Create a tensor given its raw representation.
    pub const fn from_raw(raw: T) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    /// Return a reference to the tensor's raw representation.
    pub const fn as_raw(&self) -> &T {
        &self.raw
    }

    /// Consume the tensor returning its raw representation.
    pub fn into_raw(self) -> T {
        self.raw
    }
}

impl<T, E, Ops> Tensor<T, E, Ops>
where
    Ops: ML<Repr<E> = T>,
{
    /// Create a tensor given its shape and data.
    ///
    /// The order of the elements in `data` is in increasing order of the last axis, then the second
    /// last, and so on.
    pub fn new(shape: &[NonZeroUsize], data: &[E]) -> Self
    where
        E: Elem,
    {
        Self {
            raw: Ops::new(shape, data),
            _marker: PhantomData,
        }
    }

    /// Create a scalar holding the given value.
    ///
    /// This is a special tensor that has no shape.
    pub fn scalar(value: E) -> Self
    where
        E: Elem,
    {
        Self::new(&[], &[value])
    }

    /// Return the shape of the tensor.
    pub fn shape(&self) -> &[NonZeroUsize] {
        Ops::shape::<E>(&self.raw)
    }

    /// Negate all elements
    #[must_use]
    pub fn neg(&self) -> Self
    where
        E: Num,
    {
        Self {
            raw: Ops::neg::<E>(&self.raw),
            _marker: PhantomData,
        }
    }

    /// Apply exp to each element.
    #[must_use]
    pub fn exp(&self) -> Self
    where
        E: Float,
    {
        Self {
            raw: Ops::exp::<E>(&self.raw),
            _marker: PhantomData,
        }
    }

    /// Apply the natural logarithm to each element.
    #[must_use]
    pub fn ln(&self) -> Self
    where
        E: Float,
    {
        Self {
            raw: Ops::ln::<E>(&self.raw),
            _marker: PhantomData,
        }
    }

    /// Apply sigmoid to each element.
    #[must_use]
    pub fn sigmoid(&self) -> Self
    where
        E: Float,
    {
        let ones = Self::full(self.shape(), E::one());
        let mut out = self.neg().exp();
        out = &ones + &out;
        out = &ones / &out;
        out
    }

    /// Apply tanh to each element.
    #[must_use]
    pub fn tanh(&self) -> Self
    where
        E: Float,
    {
        let ones = Self::full(self.shape(), E::one());
        let twos = Self::full(self.shape(), E::one() + E::one());
        let mut out = &twos * self;
        out = out.sigmoid();
        out = &twos * &out;
        out = &out - &ones;
        out
    }

    /// Raise `self` to the power of `other`.
    ///
    /// The tensors are broadcasted to the same shape before raising if necessary.
    #[must_use]
    pub fn pow(&self, other: &Self) -> Self
    where
        E: Float,
    {
        self.broadcast(other, Ops::pow::<E>)
    }

    /// Compare 'self' with 'other', returning 1s where the elements are equal.
    ///
    /// The tensors are broadcasted to the same shape before comparing if necessary.
    pub fn eq(&self, other: &Self) -> Tensor<<Ops as ML>::Repr<bool>, bool, Ops>
    where
        E: Num,
    {
        self.broadcast(other, Ops::eq::<E>)
    }

    /// Compare 'self' with 'other', element-wise, returning 1s where the elements are equal.
    pub fn eq_elements(&self, other: &Self) -> Tensor<<Ops as ML>::Repr<bool>, bool, Ops>
    where
        E: Elem + PartialEq,
    {
        Tensor {
            raw: Ops::eq::<E>(&self.raw, &other.raw),
            _marker: PhantomData,
        }
    }

    /// Matrix multiplication generalized for two tensors.
    ///
    /// The behavior depends on the arguments in the following ways:
    /// + If both arguments are 2-D they are multiplied like conventional matrices.
    /// + If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last
    ///   two indexes and broadcast accordingly.
    /// + If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its axes.
    ///   After matrix multiplication the prepended 1 is removed.
    /// + If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its axes.
    ///   After matrix multiplication the appended 1 is removed.
    #[allow(clippy::similar_names)]
    pub fn matmul(&self, other: &Self) -> Option<Self>
    where
        E: Num,
    {
        let mut lhs_shape = self.shape().to_vec();
        let mut rhs_shape = other.shape().to_vec();
        let orig_lhs_rank = lhs_shape.len();
        let orig_rhs_rank = rhs_shape.len();
        // Can't do matrix multiplication with scalars
        if orig_lhs_rank == 0 || orig_rhs_rank == 0 {
            return None;
        }
        // If the LHS shape is (k), make it (1, k)
        if orig_lhs_rank == 1 {
            lhs_shape.insert(0, NonZeroUsize::MIN);
        }
        // If the RHS shape is (k), make it (k, 1)
        if orig_rhs_rank == 1 {
            rhs_shape.push(NonZeroUsize::MIN);
        }
        // The last axis of the LHS must match the second-to-last axis of the RHS
        if lhs_shape[lhs_shape.len() - 1] != rhs_shape[rhs_shape.len() - 2] {
            return None;
        }
        // Turn (..., m, k) into (..., m, 1, k)
        lhs_shape.insert(lhs_shape.len() - 1, NonZeroUsize::MIN);
        // Turn (..., k, n) into (..., 1, k, n)
        rhs_shape.insert(rhs_shape.len() - 2, NonZeroUsize::MIN);
        // Multiply (..., m, 1, k) with (..., 1, n, k) to get (..., m, n, k)
        let lhs = self.reshape(&lhs_shape);
        let rhs = other.reshape(&rhs_shape);
        let mul = &lhs * &rhs.transpose(rhs_shape.len() - 1, rhs_shape.len() - 2);
        // Sum the last axis to get (..., m, n, 1)
        let sum = mul.sum(&[mul.shape().len() - 1]);
        // Remove last axis
        let mut shape = {
            let s = sum.shape();
            s[..s.len() - 1].to_vec()
        };
        // Remove prepended axis if necessary
        if orig_lhs_rank == 1 {
            shape.remove(shape.len() - 2);
        }
        // Remove appended axis if necessary
        if orig_rhs_rank == 1 {
            shape.remove(shape.len() - 1);
        }
        Some(sum.reshape(&shape))
    }

    /// Reduce along the given axes by summing all elements.
    #[must_use]
    pub fn sum(&self, axes: &[usize]) -> Self
    where
        E: Num,
    {
        Self {
            raw: Ops::sum::<E>(&self.raw, axes),
            _marker: PhantomData,
        }
    }

    /// Reduce along the given axes by getting the maximum of all elements.
    #[must_use]
    pub fn max(&self, axes: &[usize]) -> Self
    where
        E: Num,
    {
        Self {
            raw: Ops::max::<E>(&self.raw, axes),
            _marker: PhantomData,
        }
    }

    /// Reshape the tensor to the given shape, keeping the number of elements unchanged.
    #[must_use]
    pub fn reshape(&self, shape: &[NonZeroUsize]) -> Self
    where
        E: Elem,
    {
        Self {
            raw: Ops::reshape::<E>(&self.raw, shape),
            _marker: PhantomData,
        }
    }

    /// Permute the tensor axes according to the given permutation.
    #[must_use]
    pub fn permute(&self, permutation: &[usize]) -> Self
    where
        E: Elem,
    {
        Self {
            raw: Ops::permute::<E>(&self.raw, permutation),
            _marker: PhantomData,
        }
    }

    /// Expand singleton axes in a tensor to a larger size.
    #[must_use]
    pub fn expand(&self, shape: &[NonZeroUsize]) -> Self
    where
        E: Num,
    {
        Self {
            raw: Ops::expand::<E>(&self.raw, shape),
            _marker: PhantomData,
        }
    }

    /// Create a tensor given its shape filled with a single value.
    pub fn full(shape: &[NonZeroUsize], value: E) -> Self
    where
        E: Num,
    {
        Self {
            raw: Ops::full(shape, value),
            _marker: PhantomData,
        }
    }

    /// Swaps 2 dimensions of the tensor without cloning its data.
    #[must_use]
    pub fn transpose(&self, axis0: usize, axis1: usize) -> Self
    where
        E: Elem,
    {
        let mut permutation: Vec<_> = (0..self.shape().len()).collect();
        permutation.swap(axis0, axis1);
        self.permute(&permutation)
    }

    /// Removes all singleton dimensions from the tensor.
    #[must_use]
    pub fn squeeze(&self) -> Self
    where
        E: Elem,
    {
        let mut shape = self.shape().to_vec();
        shape.retain(|&sz| sz != NonZeroUsize::MIN);
        self.reshape(&shape)
    }

    fn broadcast<F, TOut, EOut, OpsOut>(&self, other: &Self, op: F) -> Tensor<TOut, EOut, OpsOut>
    where
        E: Num,
        F: Fn(&T, &T) -> TOut,
    {
        // Determine which shape has more dimensions.
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let (small, large) = if lhs_shape.len() < rhs_shape.len() {
            (lhs_shape, rhs_shape)
        } else {
            (rhs_shape, lhs_shape)
        };
        // Zipping the 2 shapes in reverse order while filling in 1 for the missing dimensions.
        let mut broadcasted_shape = large.to_vec();
        for dim in 0..small.len() {
            let sm_idx = small.len() - dim - 1;
            let lg_idx = large.len() - dim - 1;
            let sm_size = small[sm_idx];
            let lg_size = large[lg_idx];
            if sm_size == NonZeroUsize::MIN {
                broadcasted_shape[lg_idx] = lg_size;
            } else if lg_size == NonZeroUsize::MIN || lg_size == sm_size {
                broadcasted_shape[lg_idx] = sm_size;
            } else {
                panic!(
                    "broadcast: incompatible shapes ({:?} and {:?})",
                    self.shape(),
                    other.shape()
                );
            }
        }
        // Expand the tensors to the same shape and apply the operation to the expanded versions.
        let lhs = Ops::expand::<E>(&self.raw, &broadcasted_shape);
        let rhs = Ops::expand::<E>(&other.raw, &broadcasted_shape);
        Tensor {
            raw: op(&lhs, &rhs),
            _marker: PhantomData,
        }
    }
}

impl<T, E, Ops> Tensor<T, E, Ops>
where
    Ops: ToCpu<Repr<E> = T>,
{
    /// Collect all elements of the tensor into a [`Vec`].
    pub fn ravel(&self) -> Vec<E>
    where
        E: Clone,
    {
        Ops::to_cpu::<E>(&self.raw).ravel()
    }
}

/// Create an array of [`NonZeroUsize`] given an array of [`usize`].
///
/// # Panics
///
/// The function panics when any of the given [`usize`] is zero.
#[must_use]
pub fn shape<const N: usize>(shape: [usize; N]) -> [NonZeroUsize; N] {
    shape.map(|x| NonZeroUsize::new(x).expect("non-zero value"))
}
