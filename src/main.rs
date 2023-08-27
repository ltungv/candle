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
        println!("T3 elem: {:?}", x);
    }

    let t4 = t3.transpose(0, 1).unwrap();
    println!("T4 Shape: {:?}", t4.layout().shape());
    println!("T4 Strides: {:?}", t4.layout().strides());
    for x in &t4 {
        println!("T4 elem: {:?}", x);
    }

    let t5 = t3.reduce(&[0], 0.0, |x, y| x + y).unwrap();
    println!("T5 Shape: {:?}", t5.layout().shape());
    println!("T5 Strides: {:?}", t5.layout().strides());
    for x in &t5 {
        println!("T5 elem: {:?}", x);
    }

    let t6 = t3.reduce(&[1], 0.0, |x, y| x + y).unwrap();
    println!("T6 Shape: {:?}", t6.layout().shape());
    println!("T6 Strides: {:?}", t6.layout().strides());
    for x in &t6 {
        println!("T6 elem: {:?}", x);
    }

    let data: Vec<_> = (0..24).map(|x| x as f32).collect();
    let t7 = Tensor::new(&data, TensorLayout::from(&[2, 3, 4])).unwrap();
    let t7 = t7.reshape(&[6, 2, 2, 1]).unwrap();
    println!("T7 Shape: {:?}", t7.layout().shape());
    println!("T7 Strides: {:?}", t7.layout().strides());
}
