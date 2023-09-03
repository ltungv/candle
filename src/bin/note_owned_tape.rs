use std::time;

use candle::{
    dataset::Sample,
    scalar::{
        ad_backward_graph, ad_backward_tape,
        ad_backward_tape_owned::{Layer, Mlp, Var},
    },
};
use rand::Rng;
use rand_distr::StandardNormal;

fn main() {
    let dataset = dataset_add(100000);
    let mut rng = rand::thread_rng();
    let distribution = StandardNormal;
    {
        let mlp = Mlp::new(vec![Layer::rand(
            &mut rng,
            &distribution,
            Var::sigmoid,
            2,
            1,
        )]);
        let start = time::Instant::now();
        let mlp_traced = mlp.trace();
        for sample in &dataset {
            let input: Vec<_> = sample.input.iter().map(|x| Var::new(*x)).collect();
            let _ = mlp_traced.forward(&input);
        }
        let duration = time::Instant::now() - start;
        println!("inference took {}ms", duration.as_millis());
    }
    // {
    //     let tape = ad_backward_tape::Tape::default();
    //     let mlp = ad_backward_tape::Mlp::new(vec![ad_backward_tape::Layer::rand(
    //         &tape,
    //         &mut rng,
    //         &distribution,
    //         ad_backward_tape::Var::sigmoid,
    //         2,
    //         1,
    //     )]);
    //     tape.mark();
    //     let start = time::Instant::now();
    //     for sample in &dataset {
    //         let input: Vec<_> = sample.input.iter().map(|x| tape.add_variable(*x)).collect();
    //         let _ = mlp.forward(&input);
    //         tape.clean();
    //     }
    //     let duration = time::Instant::now() - start;
    //     println!("inference took {}ms", duration.as_millis());
    // }
    // {
    //     let mlp = ad_backward_graph::Mlp::new(vec![ad_backward_graph::Layer::rand(
    //         &mut rng,
    //         &distribution,
    //         ad_backward_graph::Var::sigmoid,
    //         2,
    //         1,
    //     )]);
    //     let start = time::Instant::now();
    //     for sample in &dataset {
    //         let input: Vec<_> = sample
    //             .input
    //             .iter()
    //             .map(|x| ad_backward_graph::Var::new(*x))
    //             .collect();
    //         let _ = mlp.forward(&input);
    //     }
    //     let duration = time::Instant::now() - start;
    //     println!("inference took {}ms", duration.as_millis());
    // }
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
