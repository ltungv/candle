use std::{
    cell::{Cell, RefCell},
    ops::{Add, Div, Mul, Sub},
    time,
};

use rand::{seq::SliceRandom, Rng};
use rand_distr::Distribution;

use crate::Sample;

/// A node in the computation graph holding the index of the nodes it depends on and the gradients
/// of the output with respect to each of the input.
#[derive(Debug)]
struct Node {
    from: [usize; 2],
    grad: [f64; 2],
}

/// A tape recording the computation graph where each element holds the local derivatives
/// of a variable with respect to variables that it directly depends on.
#[derive(Debug, Default)]
pub struct Tape {
    nodes: RefCell<Vec<Node>>,
    marked_position: Cell<usize>,
}

impl Tape {
    /// Get the number of nodes in the tape.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Check if the tape is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Save the current size of the tape.
    pub fn mark(&self) {
        self.marked_position.set(self.len());
    }

    /// Remove all nodes with the given range from the tape
    pub fn clean(&self) {
        self.nodes.borrow_mut().drain(self.marked_position.get()..);
    }

    /// Reset the gradients of a variable to zero.
    fn zerograd(&self, index: usize) {
        let mut nodes = self.nodes.borrow_mut();
        nodes[index].grad = [0.0, 0.0];
    }

    /// Add a node to the tape and return its index.
    fn add_node(&self, from_x: usize, from_y: usize, grad_x: f64, grad_y: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        nodes.push(Node {
            grad: [grad_x, grad_y],
            from: [from_x, from_y],
        });
        index
    }

    /// Add a variable to the tape and return it. A variable created this way does not depends on
    /// any other variable.
    pub fn add_variable(&self, value: f64) -> Variable<'_> {
        let index = {
            let id = self.len();
            self.add_node(id, id, 0.0, 0.0)
        };
        Variable {
            index,
            value,
            tape: self,
        }
    }
}

/// A variable in the computation graph. Operation on variables return new variables and do not
/// mutate the original ones.
#[derive(Debug, Clone, Copy)]
pub struct Variable<'ctx> {
    pub value: f64,
    index: usize,
    tape: &'ctx Tape,
}

impl<'ctx> Add for Variable<'ctx> {
    type Output = Variable<'ctx>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
        Variable {
            value: self.value + rhs.value,
            index: self.tape.add_node(self.index, rhs.index, 1.0, 1.0),
            tape: self.tape,
        }
    }
}

impl<'ctx> Add for &Variable<'ctx> {
    type Output = Variable<'ctx>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
        Variable {
            value: self.value + rhs.value,
            index: self.tape.add_node(self.index, rhs.index, 1.0, 1.0),
            tape: self.tape,
        }
    }
}

impl<'ctx> Sub for Variable<'ctx> {
    type Output = Variable<'ctx>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
        Variable {
            value: self.value - rhs.value,
            index: self.tape.add_node(self.index, rhs.index, 1.0, -1.0),
            tape: self.tape,
        }
    }
}

impl<'ctx> Sub for &Variable<'ctx> {
    type Output = Variable<'ctx>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
        Variable {
            value: self.value - rhs.value,
            index: self.tape.add_node(self.index, rhs.index, 1.0, -1.0),
            tape: self.tape,
        }
    }
}

impl<'ctx> Mul for Variable<'ctx> {
    type Output = Variable<'ctx>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
        Variable {
            value: self.value * rhs.value,
            index: self
                .tape
                .add_node(self.index, rhs.index, rhs.value, self.value),
            tape: self.tape,
        }
    }
}

impl<'ctx> Mul for &Variable<'ctx> {
    type Output = Variable<'ctx>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
        Variable {
            value: self.value * rhs.value,
            index: self
                .tape
                .add_node(self.index, rhs.index, rhs.value, self.value),
            tape: self.tape,
        }
    }
}

impl<'ctx> Div for Variable<'ctx> {
    type Output = Variable<'ctx>;

    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
        Variable {
            value: self.value / rhs.value,
            index: self.tape.add_node(
                self.index,
                rhs.index,
                1.0 / rhs.value,
                -self.value / (rhs.value * rhs.value),
            ),
            tape: self.tape,
        }
    }
}

impl<'ctx> Div for &Variable<'ctx> {
    type Output = Variable<'ctx>;

    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
        Variable {
            value: self.value / rhs.value,
            index: self.tape.add_node(
                self.index,
                rhs.index,
                1.0 / rhs.value,
                -self.value / (rhs.value * rhs.value),
            ),
            tape: self.tape,
        }
    }
}

impl<'ctx> Variable<'ctx> {
    pub fn gradients(&self) -> Vec<f64> {
        let mut gradients = vec![0.0; self.tape.len()];
        gradients[self.index] = 1.0;
        for (idx, n) in self.tape.nodes.borrow().iter().enumerate().rev() {
            gradients[n.from[0]] += n.grad[0] * gradients[idx];
            gradients[n.from[1]] += n.grad[1] * gradients[idx];
        }
        gradients
    }

    pub fn zerograd(&self) {
        self.tape.zerograd(self.index);
    }

    pub fn learn(&mut self, gradients: &[f64], learning_rate: f64) {
        self.value -= learning_rate * gradients[self.index];
    }

    pub fn identity(&self) -> Self {
        Variable {
            value: self.value,
            index: self.tape.add_node(self.index, self.index, 1.0, 0.0),
            tape: self.tape,
        }
    }

    pub fn sigmoid(&self) -> Self {
        let exp = self.value.exp();
        let value = exp / (1.0 + exp);
        Variable {
            value,
            index: self
                .tape
                .add_node(self.index, self.index, value * (1.0 - value), 0.0),
            tape: self.tape,
        }
    }
}

/// A neuron holding a set of weights and a bias.
struct Neuron<'a> {
    bias: Variable<'a>,
    weights: Vec<Variable<'a>>,
    nonlinearity: fn(&Variable<'a>) -> Variable<'a>,
}

impl<'a> Neuron<'a> {
    /// Create a new f32 neuron with randomized weights and bias.
    pub fn rand<R, D>(
        tape: &'a Tape,
        rng: &mut R,
        distribution: D,
        nonlinearity: fn(&Variable<'a>) -> Variable<'a>,
        input_size: usize,
    ) -> Self
    where
        R: rand::Rng,
        D: Distribution<f64> + Copy,
    {
        Self {
            bias: tape.add_variable(rng.sample(distribution)),
            weights: (0..input_size)
                .map(|_| tape.add_variable(rng.sample(distribution)))
                .collect(),
            nonlinearity,
        }
    }

    /// Applies the neuron to the given input.
    pub fn forward(&self, input: &[Variable<'a>]) -> Variable<'a> {
        assert_eq!(input.len(), self.weights.len());
        (self.nonlinearity)(
            &self
                .weights
                .iter()
                .zip(input)
                .fold(self.bias, |acc, (w, x)| acc + w * x),
        )
    }

    /// Returns a list of all parameters of the neuron.
    fn parameters(&mut self) -> Vec<&mut Variable<'a>> {
        let mut params: Vec<_> = self.weights.iter_mut().collect();
        params.push(&mut self.bias);
        params
    }
}

/// A layer of neurons.
pub struct Layer<'a> {
    neurons: Vec<Neuron<'a>>,
}

impl<'a> Layer<'a> {
    /// Create a new f32 layer with randomized neurons.
    pub fn rand<R, D>(
        tape: &'a Tape,
        rng: &mut R,
        distribution: D,
        nonlinearity: fn(&Variable<'a>) -> Variable<'a>,
        input_size: usize,
        output_size: usize,
    ) -> Self
    where
        R: rand::Rng,
        D: Distribution<f64> + Copy,
    {
        Self {
            neurons: (0..output_size)
                .map(|_| Neuron::rand(tape, rng, distribution, nonlinearity, input_size))
                .collect(),
        }
    }

    /// Applies the layer to the given input.
    pub fn forward(&self, input: &[Variable<'a>]) -> Vec<Variable<'a>> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(input))
            .collect()
    }

    /// Returns a list of all parameters of the layer.
    fn parameters(&mut self) -> Vec<&mut Variable<'a>> {
        self.neurons
            .iter_mut()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }
}

/// A multi-layer perceptron holding a set of layers.
pub struct Mlp<'a> {
    layers: Vec<Layer<'a>>,
}

impl<'a> Mlp<'a> {
    /// Create a new f32 MLP with the given list of layers.
    pub fn new(layers: Vec<Layer<'a>>) -> Self {
        Self { layers }
    }

    /// Applies the MLP to the given input.
    pub fn forward(&self, input: &[Variable<'a>]) -> Vec<Variable<'a>> {
        match self.layers.split_first() {
            Some((layer, ls)) => ls
                .iter()
                .fold(layer.forward(input), |acc, layer| layer.forward(&acc)),
            None => Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train<R, const M: usize, const N: usize>(
        &mut self,
        tape: &'a Tape,
        rng: &mut R,
        dataset: &[Sample<M, N>],
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        print_interval: usize,
    ) where
        R: Rng,
    {
        tape.mark();
        let start = time::Instant::now();
        for epoch in 0..epochs {
            let mut dataset = Vec::from(dataset);
            dataset.shuffle(rng);
            for batch in dataset.chunks(batch_size) {
                let loss = self.loss_dataset(tape, batch, mse);
                let gradients = loss.gradients();
                for param in self.parameters() {
                    param.learn(&gradients, learning_rate);
                    param.zerograd();
                }
                // Clear all intermediate nodes in the tape (nodes whose index is greater
                // than the marked index).
                tape.clean();
            }
            if print_interval != 0 && (epoch + 1) % print_interval == 0 {
                let loss = self.loss_dataset(tape, &dataset, mse);
                tape.clean();
                println!("epoch: {}, loss: {}", epoch + 1, loss.value);
            }
        }
        let duration = time::Instant::now() - start;
        println!("training took {}ms", duration.as_millis());
    }

    /// Returns a list of all parameters of the MLP.
    fn parameters(&mut self) -> Vec<&mut Variable<'a>> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn loss_sample<const M: usize, const N: usize>(
        &self,
        tape: &'a Tape,
        sample: &Sample<M, N>,
        metric: fn(&'a Tape, &[Variable<'a>], &[Variable<'a>]) -> Variable<'a>,
    ) -> Variable {
        let input: Vec<_> = sample.input.iter().map(|x| tape.add_variable(*x)).collect();
        let output: Vec<_> = sample
            .output
            .iter()
            .map(|x| tape.add_variable(*x))
            .collect();
        let pred = self.forward(&input);
        (metric)(tape, &pred, &output)
    }

    fn loss_dataset<const M: usize, const N: usize>(
        &self,
        tape: &'a Tape,
        samples: &[Sample<M, N>],
        metric: fn(&'a Tape, &[Variable<'a>], &[Variable<'a>]) -> Variable<'a>,
    ) -> Variable {
        let mut loss = tape.add_variable(0.0);
        for sample in samples {
            loss = loss + self.loss_sample(tape, sample, metric);
        }
        loss / tape.add_variable(samples.len() as f64)
    }
}

fn mse<'a>(tape: &'a Tape, pred: &[Variable<'a>], output: &[Variable<'a>]) -> Variable<'a> {
    let mut loss = tape.add_variable(0.0);
    for (p, o) in pred.iter().zip(output) {
        let diff = p - o;
        loss = loss + diff * diff;
    }
    loss / tape.add_variable(output.len() as f64)
}
