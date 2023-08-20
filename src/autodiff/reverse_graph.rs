use std::{
    cell::RefCell,
    collections::VecDeque,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
    time,
};

use rand::{seq::SliceRandom, Rng};
use rand_distr::Distribution;

use crate::Sample;

#[derive(Debug, Clone)]
struct Call {
    from: [Option<Var>; 2],
    grad: [f64; 2],
}

#[derive(Debug)]
pub struct VarInner {
    value: f64,
    grad: f64,
    call: Call,
}

#[derive(Debug, Clone)]
pub struct Var {
    inner: Rc<RefCell<VarInner>>,
}

impl Add for Var {
    type Output = Var;

    fn add(self, rhs: Self) -> Self::Output {
        let value = self.inner.borrow().value + rhs.inner.borrow().value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [1.0, 1.0],
        };
        Var::from_call(value, call)
    }
}

impl Add for &Var {
    type Output = Var;

    fn add(self, rhs: Self) -> Self::Output {
        let value = self.inner.borrow().value + rhs.inner.borrow().value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [1.0, 1.0],
        };
        Var::from_call(value, call)
    }
}

impl Sub for Var {
    type Output = Var;

    fn sub(self, rhs: Self) -> Self::Output {
        let value = self.inner.borrow().value - rhs.inner.borrow().value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [1.0, -1.0],
        };
        Var::from_call(value, call)
    }
}

impl Sub for &Var {
    type Output = Var;

    fn sub(self, rhs: Self) -> Self::Output {
        let value = self.inner.borrow().value - rhs.inner.borrow().value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [1.0, -1.0],
        };
        Var::from_call(value, call)
    }
}

impl Mul for Var {
    type Output = Var;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_value = self.inner.borrow().value;
        let rhs_value = rhs.inner.borrow().value;
        let value = lhs_value * rhs_value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [rhs_value, lhs_value],
        };
        Var::from_call(value, call)
    }
}

impl Mul for &Var {
    type Output = Var;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_value = self.inner.borrow().value;
        let rhs_value = rhs.inner.borrow().value;
        let value = lhs_value * rhs_value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [rhs_value, lhs_value],
        };
        Var::from_call(value, call)
    }
}

impl Div for Var {
    type Output = Var;

    fn div(self, rhs: Self) -> Self::Output {
        let lhs_value = self.inner.borrow().value;
        let rhs_value = rhs.inner.borrow().value;
        let value = lhs_value / rhs_value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [1.0 / rhs_value, -lhs_value / (rhs_value * rhs_value)],
        };
        Var::from_call(value, call)
    }
}

impl Div for &Var {
    type Output = Var;

    fn div(self, rhs: Self) -> Self::Output {
        let lhs_value = self.inner.borrow().value;
        let rhs_value = rhs.inner.borrow().value;
        let value = lhs_value / rhs_value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [1.0 / rhs_value, -lhs_value / (rhs_value * rhs_value)],
        };
        Var::from_call(value, call)
    }
}

impl IntoIterator for Var {
    type Item = Var;
    type IntoIter = VarIterator;

    fn into_iter(self) -> Self::IntoIter {
        VarIterator::new(self)
    }
}

impl IntoIterator for &Var {
    type Item = Var;
    type IntoIter = VarIterator;

    fn into_iter(self) -> Self::IntoIter {
        VarIterator::new(self.clone())
    }
}

impl Var {
    pub fn new(value: f64) -> Self {
        let call = Call {
            from: [None, None],
            grad: [0.0, 0.0],
        };
        Self::from_call(value, call)
    }

    pub fn value(&self) -> f64 {
        self.inner.borrow().value
    }

    pub fn grad(&self) -> f64 {
        self.inner.borrow().grad
    }

    pub fn backward(&self) {
        self.inner.borrow_mut().grad = 1.0;
        for var in self {
            var.accumulate_grad();
        }
    }

    pub fn learn(&self, rate: f64) {
        let mut inner = self.inner.borrow_mut();
        inner.value -= rate * inner.grad;
        inner.grad = 0.0;
    }

    pub fn identity(&self) -> Self {
        let value = self.inner.borrow().value;
        let call = Call {
            from: [Some(self.clone()), None],
            grad: [1.0, 0.0],
        };
        Var::from_call(value, call)
    }

    pub fn sigmoid(&self) -> Self {
        let exp = self.inner.borrow().value.exp();
        let value = exp / (1.0 + exp);
        let call = Call {
            from: [Some(self.clone()), None],
            grad: [value * (1.0 - value), 0.0],
        };
        Var::from_call(value, call)
    }

    fn accumulate_grad(&self) {
        let inner = self.inner.borrow();
        for (parent, grad) in inner.call.from.iter().zip(&inner.call.grad) {
            if let Some(p) = parent {
                let mut parent_inner = p.inner.borrow_mut();
                parent_inner.grad += inner.grad * grad;
            }
        }
    }

    fn from_call(value: f64, call: Call) -> Self {
        let inner = Rc::new(RefCell::new(VarInner {
            value,
            grad: 0.0,
            call,
        }));
        Var { inner }
    }
}

pub struct VarIterator {
    deque: VecDeque<Var>,
}

impl VarIterator {
    pub fn new(var: Var) -> Self {
        let mut deque = VecDeque::new();
        deque.push_back(var);
        Self { deque }
    }
}

impl Iterator for VarIterator {
    type Item = Var;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.deque.pop_front();
        if let Some(it) = &item {
            let it = it.inner.borrow();
            for parent in it.call.from.iter().flatten() {
                self.deque.push_back(parent.clone());
            }
        }
        item
    }
}

pub struct Neuron {
    bias: Var,
    weights: Vec<Var>,
    activation: fn(&Var) -> Var,
}

impl Neuron {
    pub fn rand<R, D>(
        rng: &mut R,
        distribution: &D,
        activation: fn(&Var) -> Var,
        input_size: usize,
    ) -> Self
    where
        R: Rng,
        D: Distribution<f64>,
    {
        let bias = Var::new(rng.gen());
        let weights = (0..input_size)
            .map(|_| Var::new(rng.sample(distribution)))
            .collect();
        Self {
            bias,
            weights,
            activation,
        }
    }

    pub fn forward(&self, input: &[Var]) -> Var {
        assert_eq!(input.len(), self.weights.len());
        (self.activation)(
            &self
                .weights
                .iter()
                .zip(input)
                .fold(self.bias.clone(), |acc, (w, x)| acc + w * x),
        )
    }

    fn parameters(&self) -> Vec<Var> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn rand<R, D>(
        rng: &mut R,
        distribution: &D,
        activation: fn(&Var) -> Var,
        input_size: usize,
        output_size: usize,
    ) -> Self
    where
        R: Rng,
        D: Distribution<f64>,
    {
        let neurons = (0..output_size)
            .map(|_| Neuron::rand(rng, distribution, activation, input_size))
            .collect();
        Self { neurons }
    }

    pub fn forward(&self, input: &[Var]) -> Vec<Var> {
        self.neurons.iter().map(|n| n.forward(input)).collect()
    }

    pub fn parameters(&self) -> Vec<Var> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}
pub struct Mlp {
    layers: Vec<Layer>,
}

impl Mlp {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn forward(&self, input: &[Var]) -> Vec<Var> {
        match self.layers.split_first() {
            Some((layer, ls)) => ls
                .iter()
                .fold(layer.forward(input), |acc, layer| layer.forward(&acc)),
            None => Vec::new(),
        }
    }

    pub fn train<R, const M: usize, const N: usize>(
        &self,
        rng: &mut R,
        dataset: &[Sample<M, N>],
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        print_interval: usize,
    ) where
        R: Rng,
    {
        let start = time::Instant::now();
        let parameters = self.parameters();
        for epoch in 0..epochs {
            let mut dataset = Vec::from(dataset);
            dataset.shuffle(rng);
            for batch in dataset.chunks(batch_size) {
                let loss = self.loss_dataset(batch, mse);
                loss.backward();
                for param in &parameters {
                    param.learn(learning_rate);
                }
            }
            if print_interval != 0 && (epoch + 1) % print_interval == 0 {
                let loss = self.loss_dataset(&dataset, mse);
                println!("epoch: {}, loss: {}", epoch + 1, loss.value());
            }
        }
        let duration = time::Instant::now() - start;
        println!("training took {}ms", duration.as_millis());
    }

    fn parameters(&self) -> Vec<Var> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn loss_sample<const M: usize, const N: usize>(
        &self,
        sample: &Sample<M, N>,
        metric: fn(&[Var], &[Var]) -> Var,
    ) -> Var {
        let input: Vec<_> = sample.input.iter().map(|x| Var::new(*x)).collect();
        let output: Vec<_> = sample.output.iter().map(|x| Var::new(*x)).collect();
        let pred = self.forward(&input);
        (metric)(&pred, &output)
    }

    fn loss_dataset<const M: usize, const N: usize>(
        &self,
        samples: &[Sample<M, N>],
        metric: fn(&[Var], &[Var]) -> Var,
    ) -> Var {
        let mut loss = Var::new(0.0);
        for sample in samples {
            loss = &loss + &self.loss_sample(sample, metric);
        }
        &loss / &Var::new(samples.len() as f64)
    }
}

fn mse(pred: &[Var], output: &[Var]) -> Var {
    let mut loss = Var::new(0.0);
    for (p, o) in pred.iter().zip(output) {
        let diff = p - o;
        loss = &loss + &(&diff * &diff);
    }
    &loss / &Var::new(output.len() as f64)
}
