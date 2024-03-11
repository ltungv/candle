use candle::{
    dataset::VectorMapping,
    scalar::{ad_backward_graph, ad_backward_tape},
};
use rand_distr::StandardNormal;

fn main() {
    let mut rng = rand::thread_rng();
    let distribution = StandardNormal;
    let dataset = vec![
        VectorMapping {
            input: [0.0, 0.0],
            output: [0.0],
        },
        VectorMapping {
            input: [0.0, 1.0],
            output: [0.0],
        },
        VectorMapping {
            input: [1.0, 0.0],
            output: [0.0],
        },
        VectorMapping {
            input: [1.0, 1.0],
            output: [1.0],
        },
    ];
    {
        let tape = ad_backward_tape::Tape::default();
        let mut mlp = ad_backward_tape::Mlp::new(vec![ad_backward_tape::Layer::rand(
            &tape,
            &mut rng,
            &distribution,
            ad_backward_tape::Var::sigmoid,
            2,
            1,
        )]);
        mlp.train(&tape, &mut rng, &dataset, 100000, 4, 0.1, 20000);
        for sample in &dataset {
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
            ad_backward_graph::Var::sigmoid,
            2,
            1,
        )]);
        mlp.train(&mut rng, &dataset, 100000, 4, 0.1, 20000);
        for sample in &dataset {
            let x1 = ad_backward_graph::Var::new(sample.input[0]);
            let x2 = ad_backward_graph::Var::new(sample.input[1]);
            let z = mlp.forward(&[x1, x2]);
            println!("pred: {}", z[0].value());
            println!("real: {}", sample.output[0]);
            println!("================")
        }
    }
}
