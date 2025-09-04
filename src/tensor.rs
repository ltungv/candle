//! An N-dimensional array.

pub mod cpu;
pub mod ops;
pub mod typ;

use std::{fmt, iter, marker::PhantomData, num::NonZeroUsize};

use ops::{ToCpu, ML};

use crate::{
    autodiff::Lift,
    tensor::typ::{Elem, Float, Num},
};

/// An alias for a tensor on the CPU having elements of type `E`.
pub type Cpu<E> = Tensor<cpu::Tensor<E>, E, cpu::TensorOps>;

/// A high-level tensor representation.
///
/// All methods on this struct are delegated to those in [`ML`], enabling auto-differentiation and
/// support for multiple hardwares. For convinience, operations are broadcasted by default, and
/// traits from [`std`] are implemented to overloads common operations for a numeric/container type.
pub struct Tensor<T, E, Ops> {
    raw: T,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> std::ops::Add<Self> for Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        self.broadcast(&other, Ops::add::<E>)
    }
}

impl<T, E, Ops> std::ops::Add<&Self> for Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Self;

    fn add(self, other: &Self) -> Self::Output {
        self.broadcast(other, Ops::add::<E>)
    }
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

impl<T, E, Ops> std::ops::Add<Tensor<T, E, Ops>> for &Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn add(self, other: Tensor<T, E, Ops>) -> Self::Output {
        self.broadcast(&other, Ops::add::<E>)
    }
}

impl<T, E, Ops> std::ops::Sub<Self> for Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.broadcast(&other, Ops::sub::<E>)
    }
}

impl<T, E, Ops> std::ops::Sub<&Self> for Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Self;

    fn sub(self, other: &Self) -> Self::Output {
        self.broadcast(other, Ops::sub::<E>)
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

impl<T, E, Ops> std::ops::Sub<Tensor<T, E, Ops>> for &Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn sub(self, other: Tensor<T, E, Ops>) -> Self::Output {
        self.broadcast(&other, Ops::sub::<E>)
    }
}

impl<T, E, Ops> std::ops::Mul<Self> for Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        self.broadcast(&other, Ops::mul::<E>)
    }
}

impl<T, E, Ops> std::ops::Mul<&Self> for Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Self;

    fn mul(self, other: &Self) -> Self::Output {
        self.broadcast(other, Ops::mul::<E>)
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

impl<T, E, Ops> std::ops::Mul<Tensor<T, E, Ops>> for &Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn mul(self, other: Tensor<T, E, Ops>) -> Self::Output {
        self.broadcast(&other, Ops::mul::<E>)
    }
}

impl<T, E, Ops> std::ops::Div<Self> for Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        self.broadcast(&other, Ops::div::<E>)
    }
}

impl<T, E, Ops> std::ops::Div<&Self> for Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Self;

    fn div(self, other: &Self) -> Self::Output {
        self.broadcast(other, Ops::div::<E>)
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

impl<T, E, Ops> std::ops::Div<Tensor<T, E, Ops>> for &Tensor<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn div(self, other: Tensor<T, E, Ops>) -> Self::Output {
        self.broadcast(&other, Ops::div::<E>)
    }
}

impl<T, E, Ops> fmt::Debug for Tensor<T, E, Ops>
where
    E: Clone + fmt::Debug,
    Ops: ToCpu<Repr<E> = T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", Ops::to_cpu::<E>(&self.raw).ravel())
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

    /// Create a tensor filled with zeroes of the given shape.
    pub fn zeroes(shape: &[NonZeroUsize]) -> Self
    where
        E: Num,
    {
        Self::full(shape, E::ZERO)
    }

    /// Create a tensor filled with ones of the given shape.
    pub fn ones(shape: &[NonZeroUsize]) -> Self
    where
        E: Num,
    {
        Self::full(shape, E::ONE)
    }

    /// Create a tensor of `num` evenly spaced elements samed from the interval `[start, stop]`.
    #[allow(clippy::similar_names)]
    pub fn linspace(start: E, stop: E, num: u16) -> Self
    where
        E: Float + From<u16>,
    {
        let step = if num > 1 {
            (stop - start.clone()) / From::from(num - 1)
        } else {
            E::ZERO
        };
        let mut data = Vec::with_capacity(num.into());
        let mut point = start;
        for _i in 0..num {
            data.push(point.clone());
            point = point.clone() + step.clone();
        }
        Self::new(&shape([num.into()]), &data)
    }

    /// Return the shape of the tensor.
    pub fn shape(&self) -> &[NonZeroUsize] {
        Ops::shape::<E>(&self.raw)
    }

    /// Lift this tensor in a Reverse AD calculation. No derivatives are calculated for this tensor, it is
    /// treated as a constant.
    pub fn lift<L>(&self) -> Tensor<L, E, L::Ops<Ops>>
    where
        L: Lift<T>,
        T: Clone,
        Ops: 'static,
    {
        L::lift(self)
    }

    /// Create a tensor filled with zeroes of the given shape.
    #[must_use]
    pub fn zeroes_like(&self) -> Self
    where
        E: Num,
    {
        Self::zeroes(self.shape())
    }

    /// Create a tensor filled with zeroes of the given shape.
    #[must_use]
    pub fn ones_like(&self) -> Self
    where
        E: Num,
    {
        Self::ones(self.shape())
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
        let ones = self.ones_like();
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
        let ones = self.ones_like();
        let twos = &ones + &ones;
        let sigmoid = (&twos * self).sigmoid();
        twos * sigmoid - ones
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
    ///
    /// # Panics
    ///
    /// Panic when either of the arguments is a scalar, or if their inner axes don't match
    #[allow(clippy::similar_names)]
    #[must_use]
    pub fn matmul(&self, other: &Self) -> Self
    where
        E: Num,
    {
        let (lhs_sh, rhs_sh) = (self.shape(), other.shape());
        let (lhs_nd, rhs_nd) = (lhs_sh.len(), rhs_sh.len());
        assert!(lhs_nd > 0, "matmul: lhs has no axis");
        assert!(rhs_nd > 0, "matmul: rhs has no axis");

        let (lhs_n, rhs_n) = (lhs_nd - 1, rhs_nd.saturating_sub(2));
        {
            let lhs = lhs_sh[lhs_n];
            let rhs = rhs_sh[rhs_n];
            assert_eq!(lhs, rhs, "matmul: mismatched axes ({lhs} != {rhs})");
        }

        let prod = if rhs_nd == 1 {
            // (.., M, N) .* (N) -> (..., M, N)
            self * other
        } else if lhs_nd == 1 {
            // (N) -> (N, 1)
            let lhs = self.reshape(&[lhs_sh, &[NonZeroUsize::MIN]].concat());
            // (N, 1) .* (..., N, O) -> (..., N, O)
            let prod = lhs * other;
            // (..., N, O) -> (..., O, N)
            let prod_nd = prod.shape().len();
            prod.transpose(prod_nd - 1, prod_nd - 2)
        } else {
            let lhs = self
                // (..., M, N) -> (..., M, 1, N)
                .reshape(&[&lhs_sh[..lhs_n], &[NonZeroUsize::MIN, lhs_sh[lhs_n]]].concat());

            let rhs = other
                // (..., N, O) -> (..., 1, N, O)
                .reshape(&[&rhs_sh[..rhs_n], &[NonZeroUsize::MIN], &rhs_sh[rhs_n..]].concat())
                // (..., 1, N, O) -> (..., 1, O, N)
                .transpose(rhs_nd, rhs_nd - 1);

            // (..., M, 1, N) .* (..., 1, O, N) -> (..., M, O, N)
            lhs * rhs
        };

        let prod_sh = prod.shape();
        let sum_n = prod_sh.len() - 1;

        // If either of the tensors is 1D:
        // (..., M, N) -> (..., M, 1)
        // (..., O, N) -> (..., O, 1)
        //
        // Otherwise:
        // (..., M, O, N) -> (..., M, O, 1)
        let sum = prod.sum(&[sum_n]);

        // If either of the tensors is 1D:
        // (..., M, 1) -> (..., M)
        // (..., O, 1) -> (..., O)
        //
        // Otherwise:
        // (..., M, O, 1) -> (..., M, O)
        sum.reshape(&prod_sh[..sum_n])
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

    /// Swaps 2 axes of the tensor without cloning its data.
    #[must_use]
    pub fn transpose(&self, axis0: usize, axis1: usize) -> Self
    where
        E: Elem,
    {
        let mut permutation: Vec<_> = (0..self.shape().len()).collect();
        permutation.swap(axis0, axis1);
        self.permute(&permutation)
    }

    /// Removes all singleton axes from the tensor.
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
        // Determine which shape has less axes, making it the lhs argument
        let (lhs, rhs, swapped) = if self.shape().len() < other.shape().len() {
            (self, other, false)
        } else {
            (other, self, true)
        };

        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        let lhs_nd = lhs_shape.len();
        let rhs_nd = rhs_shape.len();

        let mut lhs_extended_shape = Vec::with_capacity(rhs_nd);
        let mut broadcasted_shape = Vec::with_capacity(rhs_nd);

        let rhs_shape_iter = rhs_shape.iter().copied();
        let lhs_shape_iter =
            iter::repeat_n(NonZeroUsize::MIN, rhs_nd - lhs_nd).chain(lhs_shape.iter().copied());

        for (lhs_sz, rhs_sz) in lhs_shape_iter.zip(rhs_shape_iter) {
            lhs_extended_shape.push(lhs_sz);
            if lhs_sz == NonZeroUsize::MIN {
                broadcasted_shape.push(rhs_sz);
            } else if rhs_sz == NonZeroUsize::MIN || rhs_sz == lhs_sz {
                broadcasted_shape.push(lhs_sz);
            } else {
                panic!("broadcast: incompatible shapes ({lhs_shape:?} and {rhs_shape:?})",);
            }
        }

        // Expand the tensors to the same shape and apply the operation to the expanded versions.
        let lhs = lhs.reshape(&lhs_extended_shape).expand(&broadcasted_shape);
        let rhs = rhs.expand(&broadcasted_shape);
        Tensor {
            raw: if swapped {
                op(&rhs.raw, &lhs.raw)
            } else {
                op(&lhs.raw, &rhs.raw)
            },
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
