use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Sub},
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

    /// Remove all nodes with the given range from the tape
    pub fn clear(&self) {
        self.nodes.borrow_mut().clear();
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

    pub fn relu(&self) -> Self {
        let value = if self.value > 0.0 { self.value } else { 0.0 };
        Variable {
            value,
            index: self.tape.add_node(self.index, self.index, value, 0.0),
            tape: self.tape,
        }
    }
}

/// A unary operation on variable.
type UnaryOp<T> = fn(&T) -> T;

struct NeuronVariables<'a> {
    bias: Variable<'a>,
    weights: Vec<Variable<'a>>,
}

/// A neuron holding a set of weights and a bias.
struct Neuron<'a> {
    bias: f64,
    weights: Vec<f64>,
    variables: Option<NeuronVariables<'a>>,
    nonlinearity: UnaryOp<Variable<'a>>,
}

impl<'a> Neuron<'a> {
    /// Create a new f32 neuron with randomized weights and bias.
    pub fn new<R, D>(
        input_size: usize,
        nonlinearity: UnaryOp<Variable<'a>>,
        rng: &mut R,
        distribution: D,
    ) -> Self
    where
        R: rand::Rng,
        D: Distribution<f64> + Copy,
    {
        Self {
            bias: rng.sample(distribution),
            weights: (0..input_size).map(|_| rng.sample(distribution)).collect(),
            variables: None,
            nonlinearity,
        }
    }

    /// Returns a list of all parameters of the neuron.
    pub fn parameters(&mut self) -> Vec<&mut Variable<'a>> {
        let mut params = Vec::new();
        if let Some(ref mut vs) = self.variables {
            params.push(&mut vs.bias);
            vs.weights.iter_mut().for_each(|w| params.push(w));
        }
        params
    }

    pub fn update_parameters(&mut self) {
        if let Some(variables) = self.variables.take() {
            self.bias = variables.bias.value;
            self.weights = variables.weights.into_iter().map(|w| w.value).collect();
        }
    }

    pub fn clear_parameters(&mut self) {
        self.variables.take();
    }

    /// Applies the neuron to the given input.
    pub fn call(&mut self, tape: &'a Tape, input: &[Variable<'a>]) -> Variable<'a> {
        assert_eq!(input.len(), self.weights.len());
        let variables = self.variables.get_or_insert_with(|| {
            let bias = tape.add_variable(self.bias);
            let weights = self.weights.iter().map(|w| tape.add_variable(*w)).collect();
            NeuronVariables { bias, weights }
        });
        (self.nonlinearity)(
            &variables
                .weights
                .iter()
                .zip(input)
                .fold(variables.bias.identity(), |acc, (w, x)| acc + w * x),
        )
    }
}

/// A layer of neurons.
pub struct Layer<'a> {
    neurons: Vec<Neuron<'a>>,
}

impl<'a> Layer<'a> {
    /// Create a new f32 layer with randomized neurons.
    pub fn new<R, D>(
        input_size: usize,
        output_size: usize,
        nonlinearity: UnaryOp<Variable<'a>>,
        rng: &mut R,
        distribution: D,
    ) -> Self
    where
        R: rand::Rng,
        D: Distribution<f64> + Copy,
    {
        Self {
            neurons: (0..output_size)
                .map(|_| Neuron::new(input_size, nonlinearity, rng, distribution))
                .collect(),
        }
    }

    /// Returns a list of all parameters of the layer.
    pub fn parameters(&mut self) -> Vec<&mut Variable<'a>> {
        self.neurons
            .iter_mut()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }

    pub fn update_parameters(&mut self) {
        self.neurons.iter_mut().for_each(|n| n.update_parameters());
    }

    pub fn clear_parameters(&mut self) {
        self.neurons.iter_mut().for_each(|n| n.clear_parameters());
    }

    /// Applies the layer to the given input.
    pub fn call(&mut self, tape: &'a Tape, input: &[Variable<'a>]) -> Vec<Variable<'a>> {
        self.neurons
            .iter_mut()
            .map(|neuron| neuron.call(tape, input))
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

    /// Returns a list of all parameters of the MLP.
    pub fn parameters(&mut self) -> Vec<&mut Variable<'a>> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    pub fn update_parameters(&mut self) {
        self.layers.iter_mut().for_each(|l| l.update_parameters());
    }

    pub fn clear_parameters(&mut self) {
        self.layers.iter_mut().for_each(|l| l.clear_parameters());
    }

    /// Applies the MLP to the given input.
    pub fn call(&mut self, tape: &'a Tape, input: &[Variable<'a>]) -> Vec<Variable<'a>> {
        match self.layers.split_first_mut() {
            Some((layer, ls)) => ls
                .iter_mut()
                .fold(layer.call(tape, input), |acc, layer| layer.call(tape, &acc)),
            None => Vec::new(),
        }
    }
}

pub fn mse<'a, const M: usize, const N: usize>(
    tape: &'a Tape,
    mlp: &mut Mlp<'a>,
    dataset: &[Sample<M, N>],
) -> Variable<'a> {
    let mut loss = tape.add_variable(0.0);
    for sample in dataset {
        let input: Vec<_> = sample.input.iter().map(|x| tape.add_variable(*x)).collect();
        let pred = mlp.call(tape, &input);
        let mut loss_sample = tape.add_variable(0.0);
        for (p, o) in pred.iter().zip(sample.output) {
            let diff = p - &tape.add_variable(o);
            loss_sample = loss_sample + diff * diff;
        }
        loss_sample = loss_sample / tape.add_variable(sample.output.len() as f64);
        loss = loss + loss_sample;
    }
    loss / tape.add_variable(dataset.len() as f64)
}

#[allow(clippy::too_many_arguments)]
pub fn train_eval<'a, R, const M: usize, const N: usize>(
    tape: &'a Tape,
    mlp: &mut Mlp<'a>,
    rng: &mut R,
    dataset: &[Sample<M, N>],
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    print_interval: usize,
) where
    R: Rng,
{
    for epoch in 0..epochs {
        let mut dataset = Vec::from(dataset);
        dataset.shuffle(rng);
        for batch in dataset.chunks(batch_size) {
            let loss = mse(tape, mlp, batch);
            let gradients = loss.gradients();
            for param in mlp.parameters() {
                param.learn(&gradients, learning_rate);
            }
            mlp.update_parameters();
            tape.clear();
        }
        if print_interval != 0 && (epoch + 1) % print_interval == 0 {
            // Compute the loss on the entire dataset.
            let loss = mse(tape, mlp, &dataset);
            println!("Epoch: {}, Loss: {}", epoch, loss.value);
            println!("====================");
            mlp.clear_parameters();
            tape.clear();
        }
    }

    for sample in dataset {
        let input: Vec<_> = sample.input.iter().map(|x| tape.add_variable(*x)).collect();
        let pred: Vec<_> = mlp.call(tape, &input).iter().map(|x| x.value).collect();
        println!("Input: {:?}", sample.input);
        println!("Real: {:?}", sample.output);
        println!("Pred: {:?}", pred);
        println!("====================");
        mlp.clear_parameters();
        tape.clear();
    }
}
