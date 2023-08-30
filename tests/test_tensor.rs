pub mod assert;

use std::f32::consts::PI;

use assert::{assert_contiguous_tensor, assert_zipped_tensor};
use candle::tensor::Tensor;
use rand_distr::StandardNormal;

#[test]
fn test_tensor_convert_from_collections() {
    let data = [2.0, 3.0, 4.0];

    let tensor = Tensor::from(data.as_slice());
    assert_contiguous_tensor(&tensor, &data, &[3]);

    let tensor = Tensor::from(data.to_vec());
    assert_contiguous_tensor(&tensor, &data, &[3]);

    let tensor = Tensor::from(&data);
    assert_contiguous_tensor(&tensor, &data, &[3]);

    let tensor = Tensor::from(data);
    assert_contiguous_tensor(&tensor, &data, &[3]);
}

#[test]
fn test_tensor_create() {
    let data: Vec<_> = (0..2).map(|x| x as f32).collect();
    let tensor = Tensor::new(&data, &[2]).unwrap();
    assert_contiguous_tensor(&tensor, &data, &[2]);

    let data: Vec<_> = (0..6).map(|x| x as f32).collect();
    let tensor = Tensor::new(&data, &[2, 3]).unwrap();
    assert_contiguous_tensor(&tensor, &data, &[2, 3]);

    let data: Vec<_> = (0..24).map(|x| x as f32).collect();
    let tensor = Tensor::new(&data, &[2, 3, 4]).unwrap();
    assert_contiguous_tensor(&tensor, &data, &[2, 3, 4]);
}

#[test]
fn test_tensor_map() {
    let mut rng = rand::thread_rng();

    let tensor = Tensor::rand(&mut rng, StandardNormal, &[2]);
    let res = tensor.map(|x| x * PI);
    for (x, y) in tensor.into_iter().zip(res.into_iter()) {
        assert_eq!(*y, x * PI);
    }

    let tensor = Tensor::rand(&mut rng, StandardNormal, &[2, 3]);
    let res = tensor.map(|x| x + PI);
    for (x, y) in tensor.into_iter().zip(res.into_iter()) {
        assert_eq!(*y, x + PI);
    }

    let tensor = Tensor::rand(&mut rng, StandardNormal, &[2, 3, 4]);
    let res = tensor.map(|x| x / PI);
    for (x, y) in tensor.into_iter().zip(res.into_iter()) {
        assert_eq!(*y, x / PI);
    }
}

#[test]
fn test_tensor_zip() {
    let mut rng = rand::thread_rng();

    let op = |x: &f32, y: &f32| x + y;
    let t1 = Tensor::rand(&mut rng, StandardNormal, &[1]);
    let t2 = Tensor::rand(&mut rng, StandardNormal, &[3]);
    let zipped = t1.zip(&t2, op).unwrap();
    assert_zipped_tensor(&zipped, &t1, &t2, op);

    let op = |x: &f32, y: &f32| x - y;
    let t1 = Tensor::rand(&mut rng, StandardNormal, &[3]);
    let t2 = Tensor::rand(&mut rng, StandardNormal, &[1]);
    let zipped = t1.zip(&t2, op).unwrap();
    assert_zipped_tensor(&zipped, &t1, &t2, op);

    let op = |x: &f32, y: &f32| x * y;
    let t1 = Tensor::rand(&mut rng, StandardNormal, &[4, 1, 1]);
    let t2 = Tensor::rand(&mut rng, StandardNormal, &[4, 3, 2]);
    let zipped = t1.zip(&t2, op).unwrap();
    assert_zipped_tensor(&zipped, &t1, &t2, op);

    let op = |x: &f32, y: &f32| x / y;
    let t1 = Tensor::rand(&mut rng, StandardNormal, &[2]);
    let t2 = Tensor::rand(&mut rng, StandardNormal, &[4, 3, 2]);
    let zipped = t1.zip(&t2, op).unwrap();
    assert_zipped_tensor(&zipped, &t1, &t2, op);

    let op = |x: &f32, y: &f32| x + y;
    let t1 = Tensor::rand(&mut rng, StandardNormal, &[4, 3, 2]);
    let t2 = Tensor::rand(&mut rng, StandardNormal, &[4, 1, 1]);
    let zipped = t1.zip(&t2, op).unwrap();
    assert_zipped_tensor(&zipped, &t1, &t2, op);

    let op = |x: &f32, y: &f32| x - y;
    let t1 = Tensor::rand(&mut rng, StandardNormal, &[4, 3, 2]);
    let t2 = Tensor::rand(&mut rng, StandardNormal, &[2]);
    let zipped = t1.zip(&t2, op).unwrap();
    assert_zipped_tensor(&zipped, &t1, &t2, op);
}

#[test]
fn test_tensor_reduce() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let t = Tensor::new(&data, &[2, 2, 2]).unwrap();
    let t1 = t.reduce(&[0], 0.0, |x, y| x + y).unwrap();
    let t2 = t.reduce(&[1], 0.0, |x, y| x + y).unwrap();
    let t3 = t.reduce(&[2], 0.0, |x, y| x + y).unwrap();
    let t4 = t.reduce(&[0, 1], 0.0, |x, y| x + y).unwrap();
    let t5 = t.reduce(&[0, 2], 0.0, |x, y| x + y).unwrap();
    let t6 = t.reduce(&[1, 2], 0.0, |x, y| x + y).unwrap();
    let t7 = t.reduce(&[0, 1, 2], 0.0, |x, y| x + y).unwrap();

    assert_contiguous_tensor(&t1, &[4.0, 6.0, 8.0, 10.0], &[1, 2, 2]);
    assert_contiguous_tensor(&t2, &[2.0, 4.0, 10.0, 12.0], &[2, 1, 2]);
    assert_contiguous_tensor(&t3, &[1.0, 5.0, 9.0, 13.0], &[2, 2, 1]);
    assert_contiguous_tensor(&t4, &[12.0, 16.0], &[1, 1, 2]);
    assert_contiguous_tensor(&t5, &[10.0, 18.0], &[1, 2, 1]);
    assert_contiguous_tensor(&t6, &[6.0, 22.0], &[2, 1, 1]);
    assert_contiguous_tensor(&t7, &[28.0], &[1, 1, 1]);
}

#[test]
fn test_tensor_into_iter() {
    let data = [2f32, 3f32, 4f32];
    let t = Tensor::from(data);
    for (x, y) in t.into_iter().zip(&data) {
        assert_eq!(*x, *y);
    }

    let data = (0..24).map(|x| x as f32).collect::<Vec<_>>();
    let t = Tensor::new(&data, &[2, 3, 4]).unwrap();
    for (x, y) in t.into_iter().zip(&data) {
        assert_eq!(*x, *y);
    }
}
