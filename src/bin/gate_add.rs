use candle::{autodiff::reverse_graph, autodiff::reverse_tape, Sample};
use rand::{seq::SliceRandom, Rng};
use rand_distr::StandardNormal;

fn main() {
    let mut rng = rand::thread_rng();
    let distribution = StandardNormal;
    let dataset = dataset_add(1000);
    {
        let tape = reverse_tape::Tape::default();
        let mut mlp = reverse_tape::Mlp::new(vec![reverse_tape::Layer::rand(
            &tape,
            &mut rng,
            distribution,
            reverse_tape::Var::identity,
            2,
            1,
        )]);
        mlp.train(&tape, &mut rng, &dataset, 100, 100, 0.01, 20);
        for sample in dataset.choose_multiple(&mut rng, 4) {
            let x1 = tape.add_variable(sample.input[0]);
            let x2 = tape.add_variable(sample.input[1]);
            let z = mlp.forward(&[x1, x2]);
            println!("pred: {}", z[0].value);
            println!("real: {}", sample.output[0]);
            println!("================")
        }
    }
    {
        let mlp = reverse_graph::Mlp::new(vec![reverse_graph::Layer::rand(
            &mut rng,
            &distribution,
            reverse_graph::Var::identity,
            2,
            1,
        )]);
        mlp.train(&mut rng, &dataset, 100, 100, 0.01, 20);
        for sample in dataset.choose_multiple(&mut rng, 5) {
            let x1 = reverse_graph::Var::new(sample.input[0]);
            let x2 = reverse_graph::Var::new(sample.input[1]);
            let z = mlp.forward(&[x1, x2]);
            println!("pred: {}", z[0].value());
            println!("real: {}", sample.output[0]);
            println!("================")
        }
    }
}

fn dataset_add(count: usize) -> Vec<Sample<2, 1>> {
    let dist = StandardNormal;
    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let x = rng.sample(dist);
        let y = rng.sample(dist);
        let z = x + y;
        let sample = Sample {
            input: [x, y],
            output: [z],
        };
        samples.push(sample);
    }
    samples
}
