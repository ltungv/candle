use candle::tensor::{layout::TensorLayout, Tensor};

fn main() {
    let t1 = Tensor::new(&[2.0, 3.0, 4.0], TensorLayout::from(&[1, 3])).unwrap();
    println!("T1 Shape: {:?}", t1.layout().shape());
    println!("T1 Strides: {:?}", t1.layout().strides());

    let t2 = Tensor::new(&[1.0, 2.0], TensorLayout::from(&[2, 1])).unwrap();
    println!("T2 Shape: {:?}", t2.layout().shape());
    println!("T2 Strides: {:?}", t2.layout().strides());

    let t3 = t1.broadcast(&t2, |x, y| x + y).unwrap();
    println!("T3 Shape: {:?}", t3.layout().shape());
    println!("T3 Strides: {:?}", t3.layout().strides());
    for x in &t3 {
        println!("{:?}", x);
    }
    for x in t3.layout().iter_position() {
        println!("{:?}", x);
    }

    let t4 = t3.transpose();
    for x in &t4 {
        println!("{:?}", x);
    }
}
