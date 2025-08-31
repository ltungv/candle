//! An N-dimensional array.

use std::{marker::PhantomData, num::NonZeroUsize, ops};

use num::traits::bounds::LowerBounded;

use crate::ops::{ToCpu, ML};

pub mod cpu;

/// An alias for a tensor on the CPU having elements of type `E`.
pub type Cpu<E> = Tensor<cpu::Tensor<E>, E, cpu::TensorOps>;

/// A high-level tensor representation.
///
/// All methods on this struct are delegated to those in [`ML`], enabling automatic differentiation
/// and support for multiple hardwares. For convinience, operations are broadcasted by default, and
/// traits from [`std`] are implemented to overloads common operations for a numeric/container type.
#[derive(Debug)]
pub struct Tensor<T, E, Ops> {
    raw: T,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> ops::Add<Self> for &Tensor<T, E, Ops>
where
    E: Clone + ops::Add<Output = E>,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn add(self, other: Self) -> Self::Output {
        Tensor::add(self, other).expect("tensors can be broadcasted")
    }
}

impl<T, E, Ops> ops::Sub<Self> for &Tensor<T, E, Ops>
where
    E: Clone + ops::Sub<Output = E>,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn sub(self, other: Self) -> Self::Output {
        Tensor::sub(self, other).expect("tensors can be broadcasted")
    }
}

impl<T, E, Ops> ops::Mul<Self> for &Tensor<T, E, Ops>
where
    E: Clone + ops::Mul<Output = E>,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn mul(self, other: Self) -> Self::Output {
        Tensor::mul(self, other).expect("tensors can be broadcasted")
    }
}

impl<T, E, Ops> ops::Div<Self> for &Tensor<T, E, Ops>
where
    E: Clone + ops::Div<Output = E>,
    Ops: ML<Repr<E> = T>,
{
    type Output = Tensor<T, E, Ops>;

    fn div(self, other: Self) -> Self::Output {
        Tensor::div(self, other).expect("tensors can be broadcasted")
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
    pub fn new(shape: &[NonZeroUsize], data: &[E]) -> Option<Self>
    where
        E: Clone,
    {
        Some(Self {
            raw: Ops::new(shape, data)?,
            _marker: PhantomData,
        })
    }

    /// Create a scalar holding the given value.
    ///
    /// This is a special tensor that has no shape.
    pub fn scalar(value: E) -> Option<Self>
    where
        E: Clone,
    {
        Self::new(&[], &[value])
    }

    /// Create a tensor given its shape filled with a single value.
    pub fn fill(shape: &[NonZeroUsize], value: E) -> Option<Self>
    where
        E: Clone,
    {
        let tensor = Self::new(&vec![NonZeroUsize::MIN; shape.len()], &[value])?;
        tensor.expand(shape)
    }

    /// Return the shape of the tensor.
    pub fn shape(&self) -> &[NonZeroUsize] {
        Ops::shape::<E>(&self.raw)
    }

    /// Apply exp to each element.
    #[must_use]
    pub fn exp(&self) -> Self
    where
        E: num::Float,
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
        E: num::Float,
    {
        Self {
            raw: Ops::ln::<E>(&self.raw),
            _marker: PhantomData,
        }
    }

    /// Add `other` to `self`, element-wise.
    ///
    /// The tensors are broadcasted to the same shape before adding if necessary.
    #[must_use]
    pub fn add(&self, other: &Self) -> Option<Self>
    where
        E: Clone + ops::Add<Output = E>,
    {
        self.broadcast(other, |lhs, rhs| {
            let Some(out) = Ops::add::<E>(lhs, rhs) else {
                unreachable!("tensors must have the same shape");
            };
            out
        })
    }

    /// Subtract `other` from `self`, element-wise.
    ///
    /// The tensors are broadcasted to the same shape before subtracting if necessary.
    pub fn sub(&self, other: &Self) -> Option<Self>
    where
        E: Clone + ops::Sub<Output = E>,
    {
        self.broadcast(other, |lhs, rhs| {
            let Some(out) = Ops::sub::<E>(lhs, rhs) else {
                unreachable!("tensors must have the same shape");
            };
            out
        })
    }

    /// Multiply `self` by `other`, element-wise.
    ///
    /// The tensors are broadcasted to the same shape before multiplying if necessary.
    pub fn mul(&self, other: &Self) -> Option<Self>
    where
        E: Clone + ops::Mul<Output = E>,
    {
        self.broadcast(other, |lhs, rhs| {
            let Some(out) = Ops::mul::<E>(lhs, rhs) else {
                unreachable!("tensors must have the same shape");
            };
            out
        })
    }

    /// Divide `self` by `other`, element-wise.
    ///
    /// The tensors are broadcasted to the same shape before dividing if necessary.
    pub fn div(&self, other: &Self) -> Option<Self>
    where
        E: Clone + ops::Div<Output = E>,
    {
        self.broadcast(other, |lhs, rhs| {
            let Some(out) = Ops::div::<E>(lhs, rhs) else {
                unreachable!("tensors must have the same shape");
            };
            out
        })
    }

    /// Raise `self` to the power of `other`, element-wise.
    ///
    /// The tensors are broadcasted to the same shape before raising if necessary.
    pub fn pow(&self, other: &Self) -> Option<Self>
    where
        E: num::Float,
    {
        self.broadcast(other, |lhs, rhs| {
            let Some(out) = Ops::pow::<E>(lhs, rhs) else {
                unreachable!("tensors must have the same shape");
            };
            out
        })
    }

    /// Compare 'self' with 'other', element-wise, returning 1s where the elements are equal.
    ///
    /// The tensors are broadcasted to the same shape before comparing if necessary.
    pub fn eq(&self, other: &Self) -> Option<Tensor<<Ops as ML>::Repr<bool>, bool, Ops>>
    where
        E: PartialEq,
    {
        self.broadcast(other, |lhs, rhs| {
            let Some(out) = Ops::eq::<E>(lhs, rhs) else {
                unreachable!("tensors must have the same shape");
            };
            out
        })
    }

    /// Reduce along the given axes by summing all elements.
    pub fn sum(&self, axes: &[usize]) -> Option<Self>
    where
        E: Clone + num::Zero + ops::Add<Output = E>,
    {
        Some(Self {
            raw: Ops::sum::<E>(&self.raw, axes)?,
            _marker: PhantomData,
        })
    }

    /// Reduce along the given axes by getting the maximum of all elements.
    pub fn max(&self, axes: &[usize]) -> Option<Self>
    where
        E: Clone + LowerBounded + PartialOrd,
    {
        Some(Self {
            raw: Ops::max::<E>(&self.raw, axes)?,
            _marker: PhantomData,
        })
    }

    /// Reshape the tensor to the given shape, keeping the number of elements unchanged.
    pub fn reshape(&self, shape: &[NonZeroUsize]) -> Option<Self>
    where
        E: Clone,
    {
        Some(Self {
            raw: Ops::reshape::<E>(&self.raw, shape)?,
            _marker: PhantomData,
        })
    }

    /// Permute the tensor axes according to the given permutation.
    pub fn permute(&self, permutation: &[usize]) -> Option<Self>
    where
        E: Clone,
    {
        Some(Self {
            raw: Ops::permute::<E>(&self.raw, permutation)?,
            _marker: PhantomData,
        })
    }

    /// Expand singleton axes in a tensor to a larger size.
    pub fn expand(&self, shape: &[NonZeroUsize]) -> Option<Self>
    where
        E: Clone,
    {
        Some(Self {
            raw: Ops::expand::<E>(&self.raw, shape)?,
            _marker: PhantomData,
        })
    }

    /// Swaps 2 dimensions of the tensor without cloning its data.
    pub fn transpose(&self, axis0: usize, axis1: usize) -> Option<Self>
    where
        E: Clone,
    {
        let mut permutation: Vec<_> = (0..self.shape().len()).collect();
        permutation.swap(axis0, axis1);
        self.permute(&permutation)
    }

    /// Removes all singleton dimensions from the tensor.
    #[must_use]
    pub fn squeeze(&self) -> Self
    where
        E: Clone,
    {
        let mut shape = self.shape().to_vec();
        shape.retain(|&sz| sz != NonZeroUsize::MIN);
        let Some(reshaped) = self.reshape(&shape) else {
            unreachable!("tensor must be reshaped");
        };
        reshaped
    }

    /// Matrix product of two arrays.
    ///
    /// The behavior depends on the arguments in the following ways:
    /// + If both arguments are 2-D they are multiplied like conventional matrices.
    /// + If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last
    ///   two indexes and broadcast accordingly.
    /// + If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its axes.
    ///   After matrix multiplication the prepended 1 is removed.
    /// + If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its axes.
    ///   After matrix multiplication the appended 1 is removed.
    pub fn matmul(&self, other: &Self) -> Option<Self>
    where
        E: Clone + num::Zero + ops::Mul<Output = E>,
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
        let lhs = self.reshape(&lhs_shape)?;
        let rhs = other.reshape(&rhs_shape)?;
        let mul = &lhs * &rhs.transpose(rhs_shape.len() - 1, rhs_shape.len() - 2)?;
        // Sum the last axis to get (..., m, n, 1)
        let sum = mul.sum(&[mul.shape().len() - 1])?;
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
        sum.reshape(&shape)
    }

    fn broadcast<F, TOut, EOut, OpsOut>(
        &self,
        other: &Self,
        op: F,
    ) -> Option<Tensor<TOut, EOut, OpsOut>>
    where
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
                return None;
            }
        }
        // Expand the tensors to the same shape and apply the operation to the expanded versions.
        let lhs = Ops::expand::<E>(&self.raw, &broadcasted_shape)?;
        let rhs = Ops::expand::<E>(&other.raw, &broadcasted_shape)?;
        Some(Tensor {
            raw: op(&lhs, &rhs),
            _marker: PhantomData,
        })
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::{
        ops::ToCpu,
        tensor::{Cpu, Tensor},
    };

    fn shape<const N: usize>(shape: [usize; N]) -> [NonZeroUsize; N] {
        shape.map(|x| NonZeroUsize::new(x).unwrap())
    }

    fn linspace(start: f32, stop: f32, num: u16) -> Vec<f32> {
        let step = if num > 1 {
            (stop - start) / f32::from(num - 1)
        } else {
            0.0
        };
        let mut data = Vec::with_capacity(num.into());
        let mut point = start;
        for _i in 0..num {
            data.push(point);
            point += step;
        }
        data
    }

    fn assert_floats_eq(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter()
                .zip(b.iter())
                .all(|(a, b)| (a.is_nan() && b.is_nan()) || ((a - b).abs() <= f32::EPSILON)),
            "{a:?} != {b:?}"
        );
    }

    #[test]
    fn i32_tensor() {
        let c = Cpu::<i32>::scalar(2).unwrap();
        let mut t = Cpu::<i32>::new(&shape([2, 3]), &[1, 2, 3, 4, 5, 6]).unwrap();
        t = &t + &t;
        t = &c * &t;
        assert_eq!(t.shape(), &shape([2, 3]));
        assert_eq!(t.ravel(), &[4, 8, 12, 16, 20, 24]);
    }

    #[test]
    fn bool_tensor() {
        let t = Cpu::<bool>::new(&shape([2, 3]), &[true, true, false, false, true, true]).unwrap();
        let r = &t.eq(&t).unwrap();
        assert_eq!(r.shape(), &shape([2, 3]));
        assert_eq!(r.ravel(), &[true, true, true, true, true, true]);
    }

    #[test]
    fn arithmetics() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(t0: &Tensor<T, f32, I>, t1: &Tensor<T, f32, I>) {
            let r1 = t0.exp();
            assert_floats_eq(
                &r1.ravel(),
                &[
                    2.718_281_7,
                    7.389_056,
                    20.085_537,
                    54.59815,
                    148.41316,
                    403.4288,
                ],
            );

            let r2 = t0.ln();
            assert_floats_eq(
                &r2.ravel(),
                &[
                    0.0,
                    0.693_147_24,
                    1.098_612_4,
                    1.386_294_5,
                    1.609_438_1,
                    1.791_759_6,
                ],
            );

            let r3 = t0 + t1;
            assert_eq!(r3.ravel(), vec![7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);

            let r4 = t0 - t1;
            assert_eq!(r4.ravel(), vec![-5.0, -5.0, -5.0, -5.0, -5.0, -5.0]);

            let r5 = t0 / t1;
            assert_floats_eq(
                &r5.ravel(),
                &[
                    0.166_666_67,
                    0.285_714_3,
                    0.375,
                    0.444_444_45,
                    0.5,
                    0.545_454_56,
                ],
            );

            let r6 = t0 * t1;
            assert_eq!(r6.ravel(), vec![6.0, 14.0, 24.0, 36.0, 50.0, 66.0]);

            let r7 = t0.eq(t1).unwrap();
            assert_eq!(r7.ravel(), vec![false; 6]);
        }

        let shape = shape([2, 3]);
        let t0 = Cpu::<f32>::new(&shape, &linspace(1., 6., 6)).unwrap();
        let t1 = Cpu::<f32>::new(&shape, &linspace(6., 11., 6)).unwrap();
        test(&t0, &t1);
    }

    #[test]
    fn broadcasted_add() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(
            t0: &Tensor<T, f32, I>,
            t1: &Tensor<T, f32, I>,
            t2: &Tensor<T, f32, I>,
            t3: &Tensor<T, f32, I>,
        ) {
            let res = t0 + t1;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., 1., 2., 4., 5., 6.]);

            let res = t1 + t0;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., 1., 2., 4., 5., 6.]);

            let res = t0 + t2;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., 2., 4., 3., 5., 7.]);

            let res = t2 + t0;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., 2., 4., 3., 5., 7.]);

            let res = t0 + t3;
            assert_eq!(res.shape(), &shape([3, 2, 3]));
            assert_floats_eq(
                &res.ravel(),
                &[
                    0., 1., 2., 4., 5., 6., 2., 3., 4., 6., 7., 8., 4., 5., 6., 8., 9., 10.,
                ],
            );

            let res = t3 + t0;
            assert_eq!(res.shape(), &shape([3, 2, 3]));
            assert_floats_eq(
                &res.ravel(),
                &[
                    0., 1., 2., 4., 5., 6., 2., 3., 4., 6., 7., 8., 4., 5., 6., 8., 9., 10.,
                ],
            );
        }

        let t0 = Cpu::<f32>::new(&shape([2, 3]), &linspace(0., 5., 6)).unwrap();
        let t1 = Cpu::<f32>::new(&shape([2, 1]), &linspace(0., 1., 2)).unwrap();
        let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(0., 2., 3)).unwrap();
        let t3 = Cpu::<f32>::new(&shape([3, 2, 1]), &linspace(0., 5., 6)).unwrap();
        test(&t0, &t1, &t2, &t3);
    }

    #[test]
    fn broadcasted_sub() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(
            t0: &Tensor<T, f32, I>,
            t1: &Tensor<T, f32, I>,
            t2: &Tensor<T, f32, I>,
            t3: &Tensor<T, f32, I>,
        ) {
            let res = t0 - t1;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., 1., 2., 2., 3., 4.]);

            let res = t1 - t0;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., -1., -2., -2., -3., -4.]);

            let res = t0 - t2;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., 0., 0., 3., 3., 3.]);

            let res = t2 - t0;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., 0., 0., -3., -3., -3.]);

            let res = t0 - t3;
            assert_eq!(res.shape(), &shape([3, 2, 3]));
            assert_floats_eq(
                &res.ravel(),
                &[
                    0., 1., 2., 2., 3., 4., -2., -1., 0., 0., 1., 2., -4., -3., -2., -2., -1., 0.,
                ],
            );

            let res = t3 - t0;
            assert_eq!(res.shape(), &shape([3, 2, 3]));
            assert_floats_eq(
                &res.ravel(),
                &[
                    0., -1., -2., -2., -3., -4., 2., 1., 0., 0., -1., -2., 4., 3., 2., 2., 1., 0.,
                ],
            );
        }

        let t0 = Cpu::<f32>::new(&shape([2, 3]), &linspace(0., 5., 6)).unwrap();
        let t1 = Cpu::<f32>::new(&shape([2, 1]), &linspace(0., 1., 2)).unwrap();
        let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(0., 2., 3)).unwrap();
        let t3 = Cpu::<f32>::new(&shape([3, 2, 1]), &linspace(0., 5., 6)).unwrap();
        test(&t0, &t1, &t2, &t3);
    }

    #[test]
    fn broadcasted_mul() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(
            t0: &Tensor<T, f32, I>,
            t1: &Tensor<T, f32, I>,
            t2: &Tensor<T, f32, I>,
            t3: &Tensor<T, f32, I>,
        ) {
            let res = t0 * t1;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., 0., 0., 3., 4., 5.]);

            let res = t1 * t0;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., 0., 0., 3., 4., 5.]);

            let res = t0 * t2;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., 1., 4., 0., 4., 10.]);

            let res = t2 * t0;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[0., 1., 4., 0., 4., 10.]);

            let res = t0 * t3;
            assert_eq!(res.shape(), &shape([3, 2, 3]));
            assert_floats_eq(
                &res.ravel(),
                &[
                    0., 0., 0., 3., 4., 5., 0., 2., 4., 9., 12., 15., 0., 4., 8., 15., 20., 25.,
                ],
            );

            let res = t3 * t0;
            assert_eq!(res.shape(), &shape([3, 2, 3]));
            assert_floats_eq(
                &res.ravel(),
                &[
                    0., 0., 0., 3., 4., 5., 0., 2., 4., 9., 12., 15., 0., 4., 8., 15., 20., 25.,
                ],
            );
        }

        let t0 = Cpu::<f32>::new(&shape([2, 3]), &linspace(0., 5., 6)).unwrap();
        let t1 = Cpu::<f32>::new(&shape([2, 1]), &linspace(0., 1., 2)).unwrap();
        let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(0., 2., 3)).unwrap();
        let t3 = Cpu::<f32>::new(&shape([3, 2, 1]), &linspace(0., 5., 6)).unwrap();
        test(&t0, &t1, &t2, &t3);
    }

    #[test]
    fn broadcasted_div() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(
            t0: &Tensor<T, f32, I>,
            t1: &Tensor<T, f32, I>,
            t2: &Tensor<T, f32, I>,
            t3: &Tensor<T, f32, I>,
        ) {
            let res = t0 / t1;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[1., 2., 3., 2., 2.5, 3.]);

            let res = t1 / t0;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(
                &res.ravel(),
                &[1., 0.5, 0.333_333_34, 0.5, 0.4, 0.333_333_34],
            );

            let res = t0 / t2;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[1., 1., 1., 4., 2.5, 2.]);

            let res = t2 / t0;
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[1., 1., 1., 0.25, 0.4, 0.5]);

            let res = t0 / t3;
            assert_eq!(res.shape(), &shape([3, 2, 3]));
            assert_floats_eq(
                &res.ravel(),
                &[
                    1.,
                    2.,
                    3.,
                    2.,
                    2.5,
                    3.,
                    0.333_333_34,
                    0.666_666_7,
                    1.,
                    1.,
                    1.25,
                    1.5,
                    0.2,
                    0.4,
                    0.6,
                    0.666_666_7,
                    0.833_333_3,
                    1.,
                ],
            );

            let res = t3 / t0;
            assert_eq!(res.shape(), &shape([3, 2, 3]));
            assert_floats_eq(
                &res.ravel(),
                &[
                    1.,
                    0.5,
                    0.333_333_34,
                    0.5,
                    0.4,
                    0.333_333_34,
                    3.,
                    1.5,
                    1.,
                    1.,
                    0.8,
                    0.666_666_7,
                    5.,
                    2.5,
                    1.666_666_7,
                    1.5,
                    1.2,
                    1.,
                ],
            );
        }

        let t0 = Cpu::<f32>::new(&shape([2, 3]), &linspace(1., 6., 6)).unwrap();
        let t1 = Cpu::<f32>::new(&shape([2, 1]), &linspace(1., 2., 2)).unwrap();
        let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(1., 3., 3)).unwrap();
        let t3 = Cpu::<f32>::new(&shape([3, 2, 1]), &linspace(1., 6., 6)).unwrap();
        test(&t0, &t1, &t2, &t3);
    }

    #[test]
    fn broadcasted_pow() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(
            t0: &Tensor<T, f32, I>,
            t1: &Tensor<T, f32, I>,
            t2: &Tensor<T, f32, I>,
            t3: &Tensor<T, f32, I>,
        ) {
            let res = t0.pow(t1).unwrap();
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[1., 1., 1., 3., 4., 5.]);

            let res = t1.pow(t0).unwrap();
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[1., 0., 0., 1., 1., 1.]);

            let res = t0.pow(t2).unwrap();
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[1., 1., 4., 1., 4., 25.]);

            let res = t2.pow(t0).unwrap();
            assert_eq!(res.shape(), &shape([2, 3]));
            assert_floats_eq(&res.ravel(), &[1., 1., 4., 0., 1., 32.]);

            let res = t0.pow(t3).unwrap();
            assert_eq!(res.shape(), &shape([3, 2, 3]));
            assert_floats_eq(
                &res.ravel(),
                &[
                    1., 1., 1., 3., 4., 5., 0., 1., 4., 27., 64., 125., 0., 1., 16., 243., 1024.,
                    3125.,
                ],
            );

            let res = t3.pow(t0).unwrap();
            assert_eq!(res.shape(), &shape([3, 2, 3]));
            assert_floats_eq(
                &res.ravel(),
                &[
                    1., 0., 0., 1., 1., 1., 1., 2., 4., 27., 81., 243., 1., 4., 16., 125., 625.,
                    3125.,
                ],
            );
        }

        let t0 = Cpu::<f32>::new(&shape([2, 3]), &linspace(0., 5., 6)).unwrap();
        let t1 = Cpu::<f32>::new(&shape([2, 1]), &linspace(0., 1., 2)).unwrap();
        let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(0., 2., 3)).unwrap();
        let t3 = Cpu::<f32>::new(&shape([3, 2, 1]), &linspace(0., 5., 6)).unwrap();
        test(&t0, &t1, &t2, &t3);
    }

    #[test]
    fn reduce_sum() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(t: &Tensor<T, f32, I>) {
            let t1 = t.sum(&[0]).unwrap();
            assert_eq!(t1.shape(), &shape([1, 2, 2]));
            assert_floats_eq(&t1.ravel(), &[4., 6., 8., 10.]);

            let t2 = t.sum(&[1]).unwrap();
            assert_eq!(t2.shape(), &shape([2, 1, 2]));
            assert_floats_eq(&t2.ravel(), &[2., 4., 10., 12.]);

            let t3 = t.sum(&[2]).unwrap();
            assert_eq!(t3.shape(), &shape([2, 2, 1]));
            assert_floats_eq(&t3.ravel(), &[1., 5., 9., 13.]);

            let t4 = t.sum(&[0, 1]).unwrap();
            assert_eq!(t4.shape(), &shape([1, 1, 2]));
            assert_floats_eq(&t4.ravel(), &[12., 16.]);

            let t5 = t.sum(&[0, 2]).unwrap();
            assert_eq!(t5.shape(), &shape([1, 2, 1]));
            assert_floats_eq(&t5.ravel(), &[10., 18.]);

            let t6 = t.sum(&[1, 2]).unwrap();
            assert_eq!(t6.shape(), &shape([2, 1, 1]));
            assert_floats_eq(&t6.ravel(), &[6., 22.]);

            let t7 = t.sum(&[0, 1, 2]).unwrap();
            assert_eq!(t7.shape(), &shape([1, 1, 1]));
            assert_floats_eq(&t7.ravel(), &[28.]);
        }

        let t = Cpu::<f32>::new(&shape([2, 2, 2]), &linspace(0., 7., 8)).unwrap();
        test(&t);
    }

    #[test]
    fn reduce_max() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(t: &Tensor<T, f32, I>) {
            let t1 = t.max(&[0]).unwrap();
            assert_eq!(t1.shape(), &shape([1, 2, 2]));
            assert_floats_eq(&t1.ravel(), &[4., 5., 6., 7.]);

            let t2 = t.max(&[1]).unwrap();
            assert_eq!(t2.shape(), &shape([2, 1, 2]));
            assert_floats_eq(&t2.ravel(), &[2., 3., 6., 7.]);

            let t3 = t.max(&[2]).unwrap();
            assert_eq!(t3.shape(), &shape([2, 2, 1]));
            assert_floats_eq(&t3.ravel(), &[1., 3., 5., 7.]);

            let t4 = t.max(&[0, 1]).unwrap();
            assert_eq!(t4.shape(), &shape([1, 1, 2]));
            assert_floats_eq(&t4.ravel(), &[6., 7.]);

            let t5 = t.max(&[0, 2]).unwrap();
            assert_eq!(t5.shape(), &shape([1, 2, 1]));
            assert_floats_eq(&t5.ravel(), &[5., 7.]);

            let t6 = t.max(&[1, 2]).unwrap();
            assert_eq!(t6.shape(), &shape([2, 1, 1]));
            assert_floats_eq(&t6.ravel(), &[3., 7.]);

            let t7 = t.max(&[0, 1, 2]).unwrap();
            assert_eq!(t7.shape(), &shape([1, 1, 1]));
            assert_floats_eq(&t7.ravel(), &[7.]);
        }

        let t = Cpu::<f32>::new(&shape([2, 2, 2]), &linspace(0., 7., 8)).unwrap();
        test(&t);
    }

    #[test]
    fn reshape() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(t: &Tensor<T, f32, I>) {
            let data = linspace(0., 23., 24);

            let t0 = t.reshape(&shape([6, 4])).unwrap();
            assert_eq!(&t0.shape(), &shape([6, 4]));
            assert_floats_eq(&t0.ravel(), &data);

            let t1 = t.reshape(&shape([2, 12])).unwrap();
            assert_eq!(&t1.shape(), &shape([2, 12]));
            assert_floats_eq(&t1.ravel(), &data);

            let t2 = t.reshape(&shape([1, 6, 4])).unwrap();
            assert_eq!(&t2.shape(), &shape([1, 6, 4]));
            assert_floats_eq(&t2.ravel(), &data);

            let t3 = t.reshape(&shape([2, 6, 2])).unwrap();
            assert_eq!(&t3.shape(), &shape([2, 6, 2]));
            assert_floats_eq(&t3.ravel(), &data);

            let t4 = t.reshape(&shape([2, 3, 2, 2])).unwrap();
            assert_eq!(&t4.shape(), &shape([2, 3, 2, 2]));
            assert_floats_eq(&t4.ravel(), &data);

            let t5 = t.reshape(&shape([3, 8])).unwrap();
            assert_eq!(&t5.shape(), &shape([3, 8]));
            assert_floats_eq(&t5.ravel(), &data);

            let t6 = t.reshape(&shape([8, 3])).unwrap();
            assert_eq!(&t6.shape(), &shape([8, 3]));
            assert_floats_eq(&t6.ravel(), &data);
        }

        let t = Cpu::<f32>::new(&shape([2, 3, 4]), &linspace(0., 23., 24)).unwrap();
        test(&t);
    }

    #[test]
    fn permute() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(t: &Tensor<T, f32, I>) {
            let t0 = t.permute(&[0, 1, 2]).unwrap();
            assert_eq!(&t0.shape(), &shape([1, 2, 3]));
            assert_floats_eq(&t0.ravel(), &[0., 1., 2., 3., 4., 5.]);

            let t0 = t.permute(&[0, 2, 1]).unwrap();
            assert_eq!(&t0.shape(), &shape([1, 3, 2]));
            assert_floats_eq(&t0.ravel(), &[0., 3., 1., 4., 2., 5.]);

            let t0 = t.permute(&[1, 0, 2]).unwrap();
            assert_eq!(&t0.shape(), &shape([2, 1, 3]));
            assert_floats_eq(&t0.ravel(), &[0., 1., 2., 3., 4., 5.]);

            let t0 = t.permute(&[1, 2, 0]).unwrap();
            assert_eq!(&t0.shape(), &shape([2, 3, 1]));
            assert_floats_eq(&t0.ravel(), &[0., 1., 2., 3., 4., 5.]);

            let t0 = t.permute(&[2, 0, 1]).unwrap();
            assert_eq!(&t0.shape(), &shape([3, 1, 2]));
            assert_floats_eq(&t0.ravel(), &[0., 3., 1., 4., 2., 5.]);

            let t0 = t.permute(&[2, 1, 0]).unwrap();
            assert_eq!(&t0.shape(), &shape([3, 2, 1]));
            assert_floats_eq(&t0.ravel(), &[0., 3., 1., 4., 2., 5.]);
        }

        let t = Cpu::<f32>::new(&shape([1, 2, 3]), &linspace(0., 5., 6)).unwrap();
        test(&t);
    }

    #[test]
    fn expand() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(
            t0: &Tensor<T, f32, I>,
            t1: &Tensor<T, f32, I>,
            t2: &Tensor<T, f32, I>,
        ) {
            let r = t0.expand(&shape([3])).unwrap();
            assert_eq!(&r.shape(), &shape([3]));
            assert_floats_eq(&r.ravel(), &[420.69, 420.69, 420.69]);

            let r = t0.expand(&shape([3, 2])).unwrap();
            assert_eq!(&r.shape(), &shape([3, 2]));
            assert_floats_eq(
                &r.ravel(),
                &[420.69, 420.69, 420.69, 420.69, 420.69, 420.69],
            );

            let r = t1.expand(&shape([2, 3, 2])).unwrap();
            assert_eq!(&r.shape(), &shape([2, 3, 2]));
            assert_floats_eq(
                &r.ravel(),
                &[
                    420., 69., 420., 69., 420., 69., 420., 69., 420., 69., 420., 69.,
                ],
            );

            let r = t2.expand(&shape([2, 2, 3])).unwrap();
            assert_eq!(&r.shape(), &shape([2, 2, 3]));
            assert_floats_eq(
                &r.ravel(),
                &[
                    420., 420., 420., 69., 69., 69., 420., 420., 420., 69., 69., 69.,
                ],
            );
        }

        let t0 = Cpu::<f32>::new(&shape([1]), &[420.69]).unwrap();
        let t1 = Cpu::<f32>::new(&shape([1, 2]), &[420., 69.]).unwrap();
        let t2 = Cpu::<f32>::new(&shape([2, 1]), &[420., 69.]).unwrap();
        test(&t0, &t1, &t2);
    }

    #[test]
    fn transpose() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(t: &Tensor<T, f32, I>) {
            let t0 = t.transpose(0, 1).unwrap();
            assert_eq!(&t0.shape(), &shape([2, 1, 3]));
            assert_floats_eq(&t0.ravel(), &[0., 1., 2., 3., 4., 5.]);

            let t0 = t.transpose(0, 2).unwrap();
            assert_eq!(&t0.shape(), &shape([3, 2, 1]));
            assert_floats_eq(&t0.ravel(), &[0., 3., 1., 4., 2., 5.]);

            let t0 = t.transpose(1, 2).unwrap();
            assert_eq!(&t0.shape(), &shape([1, 3, 2]));
            assert_floats_eq(&t0.ravel(), &[0., 3., 1., 4., 2., 5.]);
        }

        let t = Cpu::<f32>::new(&shape([1, 2, 3]), &linspace(0., 5., 6)).unwrap();
        test(&t);
    }

    #[test]
    fn squeeze() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(
            t0: &Tensor<T, f32, I>,
            t1: &Tensor<T, f32, I>,
            t2: &Tensor<T, f32, I>,
            t3: &Tensor<T, f32, I>,
        ) {
            let r = t0.squeeze();
            assert_eq!(r.shape(), &shape([2, 3]));
            assert_floats_eq(&r.ravel(), &[0., 1., 2., 3., 4., 5.]);

            let r = t1.squeeze();
            assert_eq!(r.shape(), &shape([2, 3]));
            assert_floats_eq(&r.ravel(), &[0., 1., 2., 3., 4., 5.]);

            let r = t2.squeeze();
            assert_eq!(r.shape(), &shape([2, 3]));
            assert_floats_eq(&r.ravel(), &[0., 1., 2., 3., 4., 5.]);

            let r = t3.squeeze();
            assert_eq!(r.shape(), &shape([]));
            assert_floats_eq(&r.ravel(), &[420.69]);
        }

        let t0 = Cpu::<f32>::new(&shape([1, 2, 3]), &linspace(0., 5., 6)).unwrap();
        let t1 = Cpu::<f32>::new(&shape([2, 1, 3]), &linspace(0., 5., 6)).unwrap();
        let t2 = Cpu::<f32>::new(&shape([2, 3, 1]), &linspace(0., 5., 6)).unwrap();
        let t3 = Cpu::<f32>::new(&shape([1, 1]), &[420.69]).unwrap();
        test(&t0, &t1, &t2, &t3);
    }

    #[test]
    fn matmul() {
        fn test<T, I: ToCpu<Repr<f32> = T>>(
            t0: &Tensor<T, f32, I>,
            t1: &Tensor<T, f32, I>,
            t2: &Tensor<T, f32, I>,
            t3: &Tensor<T, f32, I>,
        ) {
            let r = t0.matmul(t0).unwrap();
            assert_eq!(r.shape(), &[]);
            assert_floats_eq(&r.ravel(), &[5.]);

            let r = t0.matmul(t1).unwrap();
            assert_eq!(r.shape(), &shape([1]));
            assert_floats_eq(&r.ravel(), &[5.]);

            let r = t0.matmul(t3).unwrap();
            assert_eq!(r.shape(), &shape([1, 3]));
            assert_floats_eq(&r.ravel(), &[15., 18., 21.]);

            let r = t2.matmul(t0).unwrap();
            assert_eq!(r.shape(), &shape([1]));
            assert_floats_eq(&r.ravel(), &[5.]);

            let r = t2.matmul(t1).unwrap();
            assert_eq!(r.shape(), &shape([1, 1]));
            assert_floats_eq(&r.ravel(), &[5.]);

            let r = t2.matmul(t3).unwrap();
            assert_eq!(r.shape(), &shape([1, 1, 3]));
            assert_floats_eq(&r.ravel(), &[15., 18., 21.]);

            let r = t3.matmul(t0).unwrap();
            assert_eq!(r.shape(), &shape([1, 3]));
            assert_floats_eq(&r.ravel(), &[5., 14., 23.]);

            let r = t3.matmul(t1).unwrap();
            assert_eq!(r.shape(), &shape([1, 3, 1]));
            assert_floats_eq(&r.ravel(), &[5., 14., 23.]);

            let r = t3.matmul(t3).unwrap();
            assert_eq!(r.shape(), &shape([1, 3, 3]));
            assert_floats_eq(&r.ravel(), &[15., 18., 21., 42., 54., 66., 69., 90., 111.]);
        }

        let t0 = Cpu::<f32>::new(&shape([3]), &linspace(0., 2., 3)).unwrap();
        let t1 = Cpu::<f32>::new(&shape([3, 1]), &linspace(0., 2., 3)).unwrap();
        let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(0., 2., 3)).unwrap();
        let t3 = Cpu::<f32>::new(&shape([1, 3, 3]), &linspace(0., 8., 9)).unwrap();
        test(&t0, &t1, &t2, &t3);
    }
}
