//! Types for differentiable functions.

use std::{marker::PhantomData, num::NonZeroUsize};

use crate::tensor::{
    ops::ML,
    typ::{Elem, Float, Num},
};

/// Unary function returning a result and a [`UnaryDiff`] for computing its derivative.
pub trait Unary<T>: UnaryDiff<T>
where
    Self: Sized,
{
    type Args: ?Sized;
    fn call(arg: &T, args: &Self::Args) -> (T, Self);
}

/// Function to compute the gradient flow of an unary function.
pub trait UnaryDiff<T> {
    /// Flow the gradient through, updating it by the local gradient w.r.t. the input.
    fn df(&self, d: &T) -> T;
}

/// Binary function returning a result and a [`BinaryDiff`] for computing its derivatives.
pub trait Binary<T>: BinaryDiff<T>
where
    Self: Sized,
{
    fn call(lhs: &T, rhs: &T) -> (T, Self);
}

/// Function to compute the gradient flow of a binary function.
pub trait BinaryDiff<T> {
    /// Flow the gradient through, updating it by the local gradient w.r.t. the LHS input.
    fn dfda(&self, d: &T) -> T;

    /// Flow the gradient through, updating it by the local gradient w.r.t. the RHS input.
    fn dfdb(&self, d: &T) -> T;
}

pub struct Exp<T, E, Ops> {
    out: T,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Unary<T> for Exp<T, E, Ops>
where
    T: Clone,
    E: Float,
    Ops: ML<Repr<E> = T>,
{
    type Args = ();

    fn call(arg: &T, (): &Self::Args) -> (T, Self) {
        let out = Ops::exp::<E>(arg);
        (
            out.clone(),
            Self {
                out,
                _marker: PhantomData,
            },
        )
    }
}

impl<T, E, Ops> UnaryDiff<T> for Exp<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn df(&self, d: &T) -> T {
        Ops::mul::<E>(d, &self.out)
    }
}

pub struct Ln<T, E, Ops> {
    arg: T,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Unary<T> for Ln<T, E, Ops>
where
    T: Clone,
    E: Float,
    Ops: ML<Repr<E> = T>,
{
    type Args = ();

    fn call(arg: &T, (): &Self::Args) -> (T, Self) {
        (
            Ops::ln::<E>(arg),
            Self {
                arg: arg.clone(),
                _marker: PhantomData,
            },
        )
    }
}

impl<T, E, Ops> UnaryDiff<T> for Ln<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn df(&self, d: &T) -> T {
        Ops::div::<E>(d, &self.arg)
    }
}

pub struct Add<E, Ops>(PhantomData<(E, Ops)>);

impl<T, E, Ops> Binary<T> for Add<E, Ops>
where
    T: Clone,
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn call(lhs: &T, rhs: &T) -> (T, Self) {
        (Ops::add::<E>(lhs, rhs), Self(PhantomData))
    }
}

impl<T, E, Ops> BinaryDiff<T> for Add<E, Ops>
where
    T: Clone,
{
    fn dfda(&self, d: &T) -> T {
        d.clone()
    }

    fn dfdb(&self, d: &T) -> T {
        d.clone()
    }
}

pub struct Sub<E, Ops>(PhantomData<(E, Ops)>);

impl<T, E, Ops> Binary<T> for Sub<E, Ops>
where
    T: Clone,
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn call(lhs: &T, rhs: &T) -> (T, Self) {
        (Ops::sub::<E>(lhs, rhs), Self(PhantomData))
    }
}

impl<T, E, Ops> BinaryDiff<T> for Sub<E, Ops>
where
    T: Clone,
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn dfda(&self, d: &T) -> T {
        d.clone()
    }

    fn dfdb(&self, d: &T) -> T {
        Ops::neg::<E>(d)
    }
}

pub struct Mul<T, E, Ops> {
    lhs: T,
    rhs: T,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Binary<T> for Mul<T, E, Ops>
where
    T: Clone,
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn call(lhs: &T, rhs: &T) -> (T, Self) {
        (
            Ops::mul::<E>(lhs, rhs),
            Self {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                _marker: PhantomData,
            },
        )
    }
}

impl<T, E, Ops> BinaryDiff<T> for Mul<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn dfda(&self, d: &T) -> T {
        Ops::mul::<E>(d, &self.rhs)
    }

    fn dfdb(&self, d: &T) -> T {
        Ops::mul::<E>(d, &self.lhs)
    }
}

pub struct Div<T, E, Ops> {
    lhs: T,
    rhs: T,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Binary<T> for Div<T, E, Ops>
where
    T: Clone,
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn call(lhs: &T, rhs: &T) -> (T, Self) {
        (
            Ops::div::<E>(lhs, rhs),
            Self {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                _marker: PhantomData,
            },
        )
    }
}

impl<T, E, Ops> BinaryDiff<T> for Div<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn dfda(&self, d: &T) -> T {
        Ops::div::<E>(d, &self.rhs)
    }

    fn dfdb(&self, d: &T) -> T {
        Ops::mul::<E>(
            d,
            &Ops::neg::<E>(&Ops::div::<E>(
                &self.lhs,
                &Ops::mul::<E>(&self.rhs, &self.rhs),
            )),
        )
    }
}

pub struct Pow<T, E, Ops> {
    lhs: T,
    rhs: T,
    out: T,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Binary<T> for Pow<T, E, Ops>
where
    T: Clone,
    E: Float,
    Ops: ML<Repr<E> = T>,
{
    fn call(lhs: &T, rhs: &T) -> (T, Self) {
        let out = Ops::pow::<E>(lhs, rhs);
        (
            out.clone(),
            Self {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                out,
                _marker: PhantomData,
            },
        )
    }
}

impl<T, E, Ops> BinaryDiff<T> for Pow<T, E, Ops>
where
    E: Float,
    Ops: ML<Repr<E> = T>,
{
    fn dfda(&self, d: &T) -> T {
        Ops::mul::<E>(
            d,
            &Ops::mul::<E>(&self.rhs, &Ops::div::<E>(&self.out, &self.lhs)),
        )
    }

    fn dfdb(&self, d: &T) -> T {
        Ops::mul::<E>(d, &Ops::mul::<E>(&self.out, &Ops::ln::<E>(&self.lhs)))
    }
}

pub struct Sum<E, Ops> {
    shape: Box<[NonZeroUsize]>,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Unary<T> for Sum<E, Ops>
where
    T: Clone,
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Args = [usize];

    fn call(arg: &T, axes: &Self::Args) -> (T, Self) {
        (
            Ops::sum::<E>(arg, axes),
            Self {
                shape: Ops::shape::<E>(arg).into(),
                _marker: PhantomData,
            },
        )
    }
}

impl<T, E, Ops> UnaryDiff<T> for Sum<E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn df(&self, d: &T) -> T {
        Ops::expand::<E>(d, &self.shape)
    }
}

pub struct Max<T, E, Ops> {
    arg: T,
    out: T,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Unary<T> for Max<T, E, Ops>
where
    T: Clone,
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Args = [usize];

    fn call(arg: &T, axes: &Self::Args) -> (T, Self) {
        let out = Ops::max::<E>(arg, axes);
        (
            out.clone(),
            Self {
                arg: arg.clone(),
                out,
                _marker: PhantomData,
            },
        )
    }
}

impl<T, E, Ops> UnaryDiff<T> for Max<T, E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn df(&self, d: &T) -> T {
        let d_shape = Ops::shape::<E>(d);
        let t_shape = Ops::shape::<E>(&self.arg);
        assert_eq!(d_shape.len(), t_shape.len());

        let reduced_axes: Box<[usize]> = t_shape
            .iter()
            .zip(d_shape.iter())
            .enumerate()
            .filter_map(|(idx, (x, y))| if x == y { None } else { Some(idx) })
            .collect();

        let eq = Ops::convert::<bool, E>(&Ops::eq::<E>(
            &self.arg,
            &Ops::expand::<E>(&self.out, t_shape),
        ));
        let eq_ratio = Ops::div::<E>(
            &eq,
            &Ops::expand::<E>(&Ops::sum::<E>(&eq, &reduced_axes), t_shape),
        );
        let d_expanded = Ops::expand::<E>(d, t_shape);
        Ops::mul::<E>(&eq_ratio, &d_expanded)
    }
}

pub struct Reshape<E, Ops> {
    shape: Box<[NonZeroUsize]>,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Unary<T> for Reshape<E, Ops>
where
    T: Clone,
    E: Elem,
    Ops: ML<Repr<E> = T>,
{
    type Args = [NonZeroUsize];

    fn call(arg: &T, shape: &Self::Args) -> (T, Self) {
        (
            Ops::reshape::<E>(arg, shape),
            Self {
                shape: shape.into(),
                _marker: PhantomData,
            },
        )
    }
}

impl<T, E, Ops> UnaryDiff<T> for Reshape<E, Ops>
where
    E: Elem,
    Ops: ML<Repr<E> = T>,
{
    fn df(&self, d: &T) -> T {
        Ops::reshape::<E>(d, &self.shape)
    }
}

pub struct Permute<E, Ops> {
    permutation: Box<[usize]>,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Unary<T> for Permute<E, Ops>
where
    E: Elem,
    Ops: ML<Repr<E> = T>,
{
    type Args = [usize];

    fn call(arg: &T, permutation: &Self::Args) -> (T, Self) {
        (
            Ops::permute::<E>(arg, permutation),
            Self {
                permutation: permutation.into(),
                _marker: PhantomData,
            },
        )
    }
}

impl<T, E, Ops> UnaryDiff<T> for Permute<E, Ops>
where
    E: Elem,
    Ops: ML<Repr<E> = T>,
{
    fn df(&self, d: &T) -> T {
        let permutation_rev: Vec<_> = {
            let mut permutation: Vec<_> = self.permutation.iter().enumerate().collect();
            permutation.sort_by_key(|&(_, axis)| *axis);
            permutation.into_iter().map(|(idx, _)| idx).collect()
        };
        Ops::permute::<E>(d, &permutation_rev)
    }
}

pub struct Expand<E, Ops> {
    shape: Box<[NonZeroUsize]>,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Unary<T> for Expand<E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    type Args = [NonZeroUsize];

    fn call(arg: &T, shape: &Self::Args) -> (T, Self) {
        (
            Ops::expand::<E>(arg, shape),
            Self {
                shape: shape.into(),
                _marker: PhantomData,
            },
        )
    }
}

impl<T, E, Ops> UnaryDiff<T> for Expand<E, Ops>
where
    E: Num,
    Ops: ML<Repr<E> = T>,
{
    fn df(&self, d: &T) -> T {
        let d_shape = Ops::shape::<E>(d);
        assert_eq!(d_shape.len(), self.shape.len());

        let expanded_axes: Box<[usize]> = self
            .shape
            .iter()
            .zip(d_shape.iter())
            .enumerate()
            .filter_map(|(idx, (x, y))| if x == y { None } else { Some(idx) })
            .collect();

        Ops::sum::<E>(d, &expanded_axes)
    }
}
