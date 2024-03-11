use candle::{
    dataset::VectorMapping,
    scalar::ad_backward_tape,
    scalar::{ad_backward_graph, ad_backward_tape_owned},
};
use rand::{seq::SliceRandom, Rng};
use rand_distr::StandardNormal;

fn main() {
    let mut rng = rand::thread_rng();
    let distribution = StandardNormal;
    let dataset = dataset_add(1000);
    {
        let tape = ad_backward_tape::Tape::default();
        let mut mlp = ad_backward_tape::Mlp::new(vec![ad_backward_tape::Layer::rand(
            &tape,
            &mut rng,
            &distribution,
            ad_backward_tape::Var::identity,
            2,
            1,
        )]);
        mlp.train(&tape, &mut rng, &dataset, 100, 100, 0.1, 20);
        for sample in dataset.choose_multiple(&mut rng, 4) {
            let x1 = tape.add_variable(sample.input[0]);
            let x2 = tape.add_variable(sample.input[1]);
            let z = mlp.forward(&[x1, x2]);
            println!("pred: {}", z[0].value());
            println!("real: {}", sample.output[0]);
            println!("================")
        }
    }
    {
        let mlp = ad_backward_graph::Mlp::new(vec![ad_backward_graph::Layer::rand(
            &mut rng,
            &distribution,
            ad_backward_graph::Var::identity,
            2,
            1,
        )]);
        mlp.train(&mut rng, &dataset, 100, 100, 0.1, 20);
        for sample in dataset.choose_multiple(&mut rng, 5) {
            let x1 = ad_backward_graph::Var::new(sample.input[0]);
            let x2 = ad_backward_graph::Var::new(sample.input[1]);
            let z = mlp.forward(&[x1, x2]);
            println!("pred: {}", z[0].value());
            println!("real: {}", sample.output[0]);
            println!("================")
        }
    }
    {
        let mut mlp = ad_backward_tape_owned::Mlp::new(vec![ad_backward_tape_owned::Layer::rand(
            &mut rng,
            &distribution,
            ad_backward_tape_owned::Var::identity,
            2,
            1,
        )]);
        mlp.trace();
        mlp.train(&mut rng, &dataset, 100, 100, 0.001, 20);
        for sample in dataset.choose_multiple(&mut rng, 5) {
            let x1 = ad_backward_tape_owned::Var::new(sample.input[0]);
            let x2 = ad_backward_tape_owned::Var::new(sample.input[1]);
            let z = mlp.forward(&[x1, x2]);
            println!("pred: {}", z[0].value);
            println!("real: {}", sample.output[0]);
            println!("================")
        }
    }
}

fn dataset_add(count: usize) -> Vec<VectorMapping<2, 1>> {
    let dist = StandardNormal;
    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let x = rng.sample(dist);
        let y = rng.sample(dist);
        let z = x + y;
        let sample = VectorMapping {
            input: [x, y],
            output: [z],
        };
        samples.push(sample);
    }
    samples
}
