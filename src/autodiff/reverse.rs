//! Reverse-mode auto-differentiation.

use std::{
    cell::{self, RefCell},
    fmt, iter,
    marker::PhantomData,
    num::NonZeroUsize,
    rc::Rc,
};

use crate::{
    autodiff::ops::{
        Add, Binary, BinaryDiff, Div, Exp, Expand, Ln, Max, Mul, Permute, Pow, Reshape, Sub, Sum,
        Unary, UnaryDiff,
    },
    tensor::{
        dtype::{Elem, Float, Num},
        ops::ML,
        Tensor,
    },
};

type AdTensor<T, E, Ops> = Tensor<Reverse<T>, E, ReverseOps<Ops>>;

/// Evaluate a function at the given primal, returning the its result and the gradient.
pub fn grad<F, T, E, Ops>(
    primal: &Tensor<T, E, Ops>,
    f: F,
) -> (Tensor<T, E, Ops>, Tensor<T, E, Ops>)
where
    F: FnOnce(AdTensor<T, E, Ops>) -> AdTensor<T, E, Ops>,
    T: 'static + Clone,
    E: Num,
    Ops: 'static + ML<Repr<E> = T>,
{
    let (out, mut grads) = gradn(&[primal], |mut ts| f(ts.swap_remove(0)));
    (out, grads.swap_remove(0))
}

/// Evaluate a function at the given primals, returning the its result and the gradient.
#[allow(clippy::type_complexity)]
pub fn gradn<F, T, E, Ops>(
    primals: &[&Tensor<T, E, Ops>],
    f: F,
) -> (Tensor<T, E, Ops>, Vec<Tensor<T, E, Ops>>)
where
    F: FnOnce(Vec<AdTensor<T, E, Ops>>) -> AdTensor<T, E, Ops>,
    T: 'static + Clone,
    E: Num,
    Ops: 'static + ML<Repr<E> = T>,
{
    let (out, pb) = PullBack::vjp(primals, f);
    let cotangent = Tensor::fill(out.shape(), E::one());
    let grad = pb.pull(&cotangent);
    (out, grad)
}

/// The back-propagation function computing a [`Vec`] of cotangents given a cotangent.
#[derive(Debug)]
pub struct PullBack<T, E, Ops> {
    tape: Rc<Tape<T>>,
    zero_primals: Vec<T>,
    result_shape: Vec<NonZeroUsize>,
    result_adjoint_index: Option<usize>,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> PullBack<T, E, Ops>
where
    Ops: 'static + ML<Repr<E> = T>,
{
    /// Evaluate a function at the given primals, building a dependency graph of evaluated functions
    /// for doing reverse-mode auto-differentiation.
    ///
    /// Return a tuple of the function's result and a [`PullBack`] object. [`PullBack::pull`] can be
    /// used to compute the Vector-Jacobian product of the function at any cotangent.
    pub fn vjp<F>(primals: &[&Tensor<T, E, Ops>], f: F) -> (Tensor<T, E, Ops>, Self)
    where
        T: Clone,
        E: Num,
        F: FnOnce(Vec<AdTensor<T, E, Ops>>) -> AdTensor<T, E, Ops>,
    {
        // Construct a cache of zero primals to be used where the VJP is not computed.
        let zero_primals: Vec<_> = primals
            .iter()
            .map(|t| Ops::fill::<E>(t.shape(), E::zero()))
            .collect();

        // Construct the variables from the given primals and add them to the tape.
        let tape = Rc::new(Tape::new());
        let vars: Vec<_> = primals
            .iter()
            .map(|&t| {
                Tensor::from_raw(Reverse::Traced(Trace::new(
                    &tape,
                    TapeNode::Var,
                    t.as_raw().clone(),
                )))
            })
            .collect();

        // Execute the function. Because [`Reverse`] is used to construct the variable, a dependency
        // graph will be created on the tape once the function returns.
        let result = f(vars);
        let result_shape = result.shape().to_vec();
        let (primal, result_adjoint_index) = match result.into_raw() {
            Reverse::Lifted(primal) => (primal, None),
            Reverse::Traced(trace) => (trace.primal, Some(trace.index)),
        };

        (
            Tensor::from_raw(primal),
            Self {
                tape: Rc::clone(&tape),
                zero_primals,
                result_shape,
                result_adjoint_index,
                _marker: PhantomData,
            },
        )
    }

    /// Take a cotangent with the same shape as the result of the originating VJP function, and
    /// return a [`Vec`] of cotangents of the same length and shapes as the VJP's primals,
    pub fn pull(&self, cotangent: &Tensor<T, E, Ops>) -> Vec<Tensor<T, E, Ops>>
    where
        T: 'static + Clone,
        E: Num,
    {
        let ts = self.result_adjoint_index.map_or_else(
            || self.zero_primals.clone(),
            |idx| self.reverse(idx, cotangent.as_raw()),
        );
        ts.into_iter().map(|raw| Tensor::from_raw(raw)).collect()
    }

    // Back-propagate from the given adjoint index.
    #[allow(clippy::similar_names)]
    fn reverse(&self, adjoint_index: usize, adjoint: &T) -> Vec<T>
    where
        T: 'static + Clone,
        E: Num,
    {
        assert_eq!(self.result_shape, Ops::shape::<E>(adjoint));
        let mut adjoints: Vec<_> = iter::repeat_with(Adjoint::<T, E, Ops>::empty)
            .take(adjoint_index + 1)
            .collect();

        adjoints[adjoint_index] = Adjoint::new(adjoint.clone());
        for i in (0..=adjoint_index).rev() {
            if let Some(adjoint) = adjoints[i].as_ref() {
                let nodes = self.tape.nodes();
                match &nodes[i] {
                    TapeNode::Var => break,
                    TapeNode::Unary(op, a) => {
                        let df = op.df(adjoint);
                        adjoints[*a].accumulate(df);
                    }
                    TapeNode::BinaryDA(op, a) => {
                        let dfda = op.dfda(adjoint);
                        adjoints[*a].accumulate(dfda);
                    }
                    TapeNode::BinaryDB(op, b) => {
                        let dfdb = op.dfdb(adjoint);
                        adjoints[*b].accumulate(dfdb);
                    }
                    TapeNode::Binary(op, a, b) => {
                        let dfda = op.dfda(adjoint);
                        let dfdb = op.dfdb(adjoint);
                        adjoints[*a].accumulate(dfda);
                        adjoints[*b].accumulate(dfdb);
                    }
                }
            }
            adjoints.pop();
        }
        assert_eq!(adjoints.len(), self.zero_primals.len());
        adjoints
            .into_iter()
            .zip(self.zero_primals.iter())
            .map(|(mut x, z)| x.take().unwrap_or_else(|| z.clone()))
            .collect()
    }
}

/// Helper type for an uninitialized adjoint.
struct Adjoint<T, E, Ops> {
    df: Option<T>,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Adjoint<T, E, Ops> {
    const fn empty() -> Self {
        Self {
            df: None,
            _marker: PhantomData,
        }
    }

    const fn new(df: T) -> Self {
        Self {
            df: Some(df),
            _marker: PhantomData,
        }
    }

    const fn take(&mut self) -> Option<T> {
        self.df.take()
    }

    const fn as_ref(&self) -> Option<&T> {
        self.df.as_ref()
    }

    fn accumulate(&mut self, df: T)
    where
        E: Num,
        Ops: ML<Repr<E> = T>,
    {
        match self.df.as_mut() {
            None => self.df = Some(df),
            Some(c) => *c = Ops::add::<E>(c, &df).expect("accumulate is infallible"),
        }
    }
}

/// Mid-level tensor operations supporting reverse-mode auto-differentiation.
#[derive(Debug)]
pub struct ReverseOps<Ops>(PhantomData<Ops>);

impl<Ops> ML for ReverseOps<Ops>
where
    Ops: 'static + ML,
{
    type Repr<E> = Reverse<Ops::Repr<E>>;

    fn new<E>(shape: &[NonZeroUsize], data: &[E]) -> Option<Self::Repr<E>>
    where
        E: Elem,
    {
        Ops::new::<E>(shape, data).map(Reverse::Lifted)
    }

    fn convert<E, EInto>(t: &Self::Repr<E>) -> Self::Repr<EInto>
    where
        E: Elem + Into<EInto>,
        EInto: Elem,
    {
        Reverse::Lifted(Ops::convert::<E, EInto>(t.primal()))
    }

    fn shape<E>(t: &Self::Repr<E>) -> &[NonZeroUsize] {
        Ops::shape::<E>(t.primal())
    }

    fn exp<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float,
    {
        t.unary(|t| Exp::<Ops::Repr<E>, E, Ops>::call(t, &()))
            .expect("exp is infallible")
    }

    fn ln<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float,
    {
        t.unary(|t| Ln::<Ops::Repr<E>, E, Ops>::call(t, &()))
            .expect("ln is infallible")
    }

    fn add<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        lhs.binary(rhs, Add::<E, Ops>::call)
    }

    fn sub<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        lhs.binary(rhs, Sub::<E, Ops>::call)
    }

    fn mul<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        lhs.binary(rhs, Mul::<Ops::Repr<E>, E, Ops>::call)
    }

    fn div<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        lhs.binary(rhs, Div::<Ops::Repr<E>, E, Ops>::call)
    }

    fn pow<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Float,
    {
        lhs.binary(rhs, Pow::<Ops::Repr<E>, E, Ops>::call)
    }

    fn eq<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<bool>>
    where
        E: Elem + PartialEq,
    {
        Ops::eq::<E>(lhs.primal(), rhs.primal()).map(Reverse::Lifted)
    }

    fn sum<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        t.unary(|t| Sum::<E, Ops>::call(t, axes))
    }

    fn max<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        t.unary(|t| Max::<Ops::Repr<E>, E, Ops>::call(t, axes))
    }

    fn reshape<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>>
    where
        E: Elem,
    {
        t.unary(|t| Reshape::<E, Ops>::call(t, shape))
    }

    fn permute<E>(t: &Self::Repr<E>, permutation: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Elem,
    {
        t.unary(|t| Permute::<E, Ops>::call(t, permutation))
    }

    fn expand<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        t.unary(|t| Expand::<E, Ops>::call(t, shape))
    }
}

/// Reverse auto-differentation support.
#[derive(Clone, Debug)]
pub enum Reverse<T> {
    /// A lifted value whose gradient won't be calculated.
    Lifted(T),

    /// A traced value tracking the dependency graph of operations.
    Traced(Trace<T>),
}

impl<T> Reverse<T> {
    const fn primal(&self) -> &T {
        match self {
            Self::Lifted(primal) => primal,
            Self::Traced(trace) => &trace.primal,
        }
    }

    fn unary<F, Grad>(&self, op: F) -> Option<Self>
    where
        F: FnOnce(&T) -> Option<(T, Grad)>,
        Grad: 'static + UnaryDiff<T>,
    {
        match self {
            Self::Lifted(primal) => {
                let (primal, _) = op(primal)?;
                Some(Self::Lifted(primal))
            }

            Self::Traced(trace) => {
                let (primal, grad) = op(&trace.primal)?;
                Some(Self::Traced(Trace::new(
                    &trace.tape,
                    TapeNode::Unary(Box::new(grad), trace.index),
                    primal,
                )))
            }
        }
    }

    fn binary<F, Grad>(&self, other: &Self, op: F) -> Option<Self>
    where
        F: FnOnce(&T, &T) -> Option<(T, Grad)>,
        Grad: 'static + BinaryDiff<T>,
    {
        match (self, other) {
            (Self::Lifted(lhs), Self::Lifted(rhs)) => {
                let (primal, _) = op(lhs, rhs)?;
                Some(Self::Lifted(primal))
            }

            (Self::Lifted(lhs), Self::Traced(rhs)) => {
                let (primal, grad) = op(lhs, &rhs.primal)?;
                Some(Self::Traced(Trace::new(
                    &rhs.tape,
                    TapeNode::BinaryDB(Box::new(grad), rhs.index),
                    primal,
                )))
            }

            (Self::Traced(lhs), Self::Lifted(rhs)) => {
                let (primal, grad) = op(&lhs.primal, rhs)?;
                Some(Self::Traced(Trace::new(
                    &lhs.tape,
                    TapeNode::BinaryDA(Box::new(grad), lhs.index),
                    primal,
                )))
            }

            (Self::Traced(lhs), Self::Traced(rhs)) => {
                assert!(Rc::ptr_eq(&lhs.tape, &rhs.tape));
                let (primal, grad) = op(&lhs.primal, &rhs.primal)?;
                Some(Self::Traced(Trace::new(
                    &lhs.tape,
                    TapeNode::Binary(Box::new(grad), lhs.index, rhs.index),
                    primal,
                )))
            }
        }
    }
}

/// Container for a primal, an index to its tangent, and a reference to the tape.
#[derive(Clone, Debug)]
pub struct Trace<T> {
    tape: Rc<Tape<T>>,
    index: usize,
    primal: T,
}

impl<T> Trace<T> {
    fn new(tape: &Rc<Tape<T>>, node: TapeNode<T>, primal: T) -> Self {
        let index = tape.push(node);
        Self {
            tape: Rc::clone(tape),
            index,
            primal,
        }
    }
}

#[derive(Debug)]
struct Tape<T> {
    nodes: RefCell<Vec<TapeNode<T>>>,
}

impl<T> Default for Tape<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Tape<T> {
    #[must_use]
    const fn new() -> Self {
        Self {
            nodes: RefCell::new(vec![]),
        }
    }

    fn push(&self, node: TapeNode<T>) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        nodes.push(node);
        index
    }

    fn nodes(&self) -> cell::Ref<'_, Vec<TapeNode<T>>> {
        self.nodes.borrow()
    }
}

enum TapeNode<T> {
    Var,
    Unary(Box<dyn UnaryDiff<T>>, usize),
    BinaryDA(Box<dyn BinaryDiff<T>>, usize),
    BinaryDB(Box<dyn BinaryDiff<T>>, usize),
    Binary(Box<dyn BinaryDiff<T>>, usize, usize),
}

impl<T> fmt::Debug for TapeNode<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var => f.debug_tuple("TapeNode::Var").finish(),

            Self::Unary(op, idx) => f
                .debug_tuple("TapeNode::Unary")
                .field(&format_args!("{op:#p}"))
                .field(idx)
                .finish(),

            Self::BinaryDA(op, idx) => f
                .debug_tuple("TapeNode::BinaryDA")
                .field(&format_args!("{op:#p}"))
                .field(idx)
                .finish(),

            Self::BinaryDB(op, idx) => f
                .debug_tuple("TapeNode::BinaryDB")
                .field(&format_args!("{op:#p}"))
                .field(idx)
                .finish(),

            Self::Binary(op, idx0, idx1) => f
                .debug_tuple("TapeNode::Binary")
                .field(&format_args!("{op:#p}"))
                .field(idx0)
                .field(idx1)
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::{autodiff::reverse::grad, tensor::Cpu};

    fn shape<const N: usize>(shape: [usize; N]) -> [NonZeroUsize; N] {
        shape.map(|x| NonZeroUsize::new(x).unwrap())
    }

    #[test]
    fn it_works() {
        let t = Cpu::<f32>::new(&shape([2, 3]), &[1., 1., 1., 1., 1., 1.]).unwrap();
        let (r, g) = grad(&t, |t| t.max(&[1]).unwrap());
        dbg!(r.ravel());
        dbg!(g.ravel());
    }
}
