use candle::{
    autodiff::{reverse_graph, reverse_tape},
    Sample,
};
use rand_distr::StandardNormal;

fn main() {
    let mut rng = rand::thread_rng();
    let distribution = StandardNormal;
    let dataset = vec![
        Sample {
            input: [0.0, 0.0],
            output: [0.0],
        },
        Sample {
            input: [0.0, 1.0],
            output: [1.0],
        },
        Sample {
            input: [1.0, 0.0],
            output: [1.0],
        },
        Sample {
            input: [1.0, 1.0],
            output: [0.0],
        },
    ];
    {
        let tape = reverse_tape::Tape::default();
        let mut mlp = reverse_tape::Mlp::new(vec![
            reverse_tape::Layer::rand(
                &tape,
                &mut rng,
                distribution,
                reverse_tape::Var::sigmoid,
                2,
                3,
            ),
            reverse_tape::Layer::rand(
                &tape,
                &mut rng,
                distribution,
                reverse_tape::Var::sigmoid,
                3,
                1,
            ),
        ]);
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
        let mlp = reverse_graph::Mlp::new(vec![
            reverse_graph::Layer::rand(&mut rng, &distribution, reverse_graph::Var::sigmoid, 2, 3),
            reverse_graph::Layer::rand(&mut rng, &distribution, reverse_graph::Var::sigmoid, 3, 1),
        ]);
        mlp.train(&mut rng, &dataset, 100000, 4, 0.1, 20000);
        for sample in &dataset {
            let x1 = reverse_graph::Var::new(sample.input[0]);
            let x2 = reverse_graph::Var::new(sample.input[1]);
            let z = mlp.forward(&[x1, x2]);
            println!("pred: {}", z[0].value());
            println!("real: {}", sample.output[0]);
            println!("================")
        }
    }
}
