pub mod assert;

use std::f32::consts::PI;

use assert::{assert_contiguous_tensor, assert_scalar_tensor};
use candle::tensor::Tensor;
use rand_distr::StandardNormal;

fn linspace(start: f32, stop: f32, num: usize) -> Vec<f32> {
    let step = (stop - start) / num as f32;
    (0..num).map(|x| start + step * x as f32).collect()
}

#[test]
fn test_tensor_broadcasted_add() {
    let t0 = Tensor::shaped(&[2, 3], &linspace(0., 6., 6)).unwrap();
    let t1 = Tensor::shaped(&[2, 1], &linspace(0., 2., 2)).unwrap();
    let t2 = Tensor::shaped(&[1, 3], &linspace(0., 3., 3)).unwrap();
    let t3 = Tensor::shaped(&[3, 2, 1], &linspace(0., 6., 6)).unwrap();

    let res = &t0 + &t1;
    assert_contiguous_tensor(&res, &[0., 1., 2., 4., 5., 6.], &[2, 3]);

    let res = &t1 + &t0;
    assert_contiguous_tensor(&res, &[0., 1., 2., 4., 5., 6.], &[2, 3]);

    let res = &t0 + &t2;
    assert_contiguous_tensor(&res, &[0., 2., 4., 3., 5., 7.], &[2, 3]);

    let res = &t2 + &t0;
    assert_contiguous_tensor(&res, &[0., 2., 4., 3., 5., 7.], &[2, 3]);

    // [[ 0, 1, 2 ]
    //  [ 3, 4, 5 ]]
    //
    // [[[ 0 ], [ 1 ]],
    //  [[ 2 ], [ 3 ]],
    //  [[ 4 ], [ 5 ]]]
    //
    // [[[ 0, 1, 2 ], [ 4, 5, 6 ]],
    //  [[ 2  3, 4 ], [ 3, 4, 5 ]],
    //  [[ 4  5, 6 ], [ 8, 9, 10 ]]]

    let res = &t0 + &t3;
    assert_contiguous_tensor(
        &res,
        &[
            0., 1., 2., 4., 5., 6., 2., 3., 4., 6., 7., 8., 4., 5., 6., 8., 9., 10.,
        ],
        &[3, 2, 3],
    );

    let res = &t3 + &t0;
    assert_contiguous_tensor(
        &res,
        &[
            0., 1., 2., 4., 5., 6., 2., 3., 4., 6., 7., 8., 4., 5., 6., 8., 9., 10.,
        ],
        &[3, 2, 3],
    );
}

#[test]
fn test_tensor_broadcasted_mul() {
    let t0 = Tensor::shaped(&[2, 3], &linspace(0., 6., 6)).unwrap();
    let t1 = Tensor::shaped(&[2, 1], &linspace(0., 2., 2)).unwrap();
    let t2 = Tensor::shaped(&[1, 3], &linspace(0., 3., 3)).unwrap();
    let t3 = Tensor::shaped(&[3, 2, 1], &linspace(0., 6., 6)).unwrap();

    let res = &t0 * &t1;
    assert_contiguous_tensor(&res, &[0., 0., 0., 3., 4., 5.], &[2, 3]);

    let res = &t1 * &t0;
    assert_contiguous_tensor(&res, &[0., 0., 0., 3., 4., 5.], &[2, 3]);

    let res = &t0 * &t2;
    assert_contiguous_tensor(&res, &[0., 1., 4., 0., 4., 10.], &[2, 3]);

    let res = &t2 * &t0;
    assert_contiguous_tensor(&res, &[0., 1., 4., 0., 4., 10.], &[2, 3]);

    // [[ 0, 1, 2 ]
    //  [ 3, 4, 5 ]]
    //
    // [[[ 0 ], [ 1 ]],
    //  [[ 2 ], [ 3 ]],
    //  [[ 4 ], [ 5 ]]]
    //
    // [[[ 0, 0, 0 ], [ 3, 4, 5 ]],
    //  [[ 0  2, 4 ], [ 9, 12, 15 ]],
    //  [[ 0  4, 8 ], [ 15, 20, 25 ]]]

    let res = &t0 * &t3;
    assert_contiguous_tensor(
        &res,
        &[
            0., 0., 0., 3., 4., 5., 0., 2., 4., 9., 12., 15., 0., 4., 8., 15., 20., 25.,
        ],
        &[3, 2, 3],
    );

    let res = &t3 * &t0;
    assert_contiguous_tensor(
        &res,
        &[
            0., 0., 0., 3., 4., 5., 0., 2., 4., 9., 12., 15., 0., 4., 8., 15., 20., 25.,
        ],
        &[3, 2, 3],
    );
}

#[test]
fn test_tensor_sum() {
    let t = Tensor::shaped(
        &[2, 2, 2],
        (0..8).map(|x| x as f32).collect::<Vec<_>>().as_slice(),
    )
    .unwrap();
    let t1 = t.sum(&[0]).unwrap();
    let t2 = t.sum(&[1]).unwrap();
    let t3 = t.sum(&[2]).unwrap();
    let t4 = t.sum(&[0, 1]).unwrap();
    let t5 = t.sum(&[0, 2]).unwrap();
    let t6 = t.sum(&[1, 2]).unwrap();
    let t7 = t.sum(&[0, 1, 2]).unwrap();

    assert_contiguous_tensor(&t1, &[4.0, 6.0, 8.0, 10.0], &[1, 2, 2]);
    assert_contiguous_tensor(&t2, &[2.0, 4.0, 10.0, 12.0], &[2, 1, 2]);
    assert_contiguous_tensor(&t3, &[1.0, 5.0, 9.0, 13.0], &[2, 2, 1]);
    assert_contiguous_tensor(&t4, &[12.0, 16.0], &[1, 1, 2]);
    assert_contiguous_tensor(&t5, &[10.0, 18.0], &[1, 2, 1]);
    assert_contiguous_tensor(&t6, &[6.0, 22.0], &[2, 1, 1]);
    assert_contiguous_tensor(&t7, &[28.0], &[1, 1, 1]);
}

#[test]
fn test_tensor_prod() {
    let t = Tensor::shaped(
        &[2, 2, 2],
        (1..9).map(|x| x as f32).collect::<Vec<_>>().as_slice(),
    )
    .unwrap();
    let t1 = t.prod(&[0]).unwrap();
    let t2 = t.prod(&[1]).unwrap();
    let t3 = t.prod(&[2]).unwrap();
    let t4 = t.prod(&[0, 1]).unwrap();
    let t5 = t.prod(&[0, 2]).unwrap();
    let t6 = t.prod(&[1, 2]).unwrap();
    let t7 = t.prod(&[0, 1, 2]).unwrap();

    assert_contiguous_tensor(&t1, &[5.0, 12.0, 21.0, 32.0], &[1, 2, 2]);
    assert_contiguous_tensor(&t2, &[3.0, 8.0, 35.0, 48.0], &[2, 1, 2]);
    assert_contiguous_tensor(&t3, &[2.0, 12.0, 30.0, 56.0], &[2, 2, 1]);
    assert_contiguous_tensor(&t4, &[105.0, 384.0], &[1, 1, 2]);
    assert_contiguous_tensor(&t5, &[60.0, 672.0], &[1, 2, 1]);
    assert_contiguous_tensor(&t6, &[24.0, 1680.0], &[2, 1, 1]);
    assert_contiguous_tensor(&t7, &[40320.0], &[1, 1, 1]);
}

#[test]
fn test_tensor_map() {
    let mut rng = rand::thread_rng();

    let tensor = Tensor::rand(&mut rng, StandardNormal, &[2]);
    let res = tensor.map(|x| x * PI);
    for (x, y) in tensor.into_iter().zip(res.into_iter()) {
        assert_eq!(y, x * PI);
    }

    let tensor = Tensor::rand(&mut rng, StandardNormal, &[2, 3]);
    let res = tensor.map(|x| x + PI);
    for (x, y) in tensor.into_iter().zip(res.into_iter()) {
        assert_eq!(y, x + PI);
    }

    let tensor = Tensor::rand(&mut rng, StandardNormal, &[2, 3, 4]);
    let res = tensor.map(|x| x / PI);
    for (x, y) in tensor.into_iter().zip(res.into_iter()) {
        assert_eq!(y, x / PI);
    }
}

#[test]
fn test_tensor_reduce() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let t = Tensor::shaped(&[2, 2, 2], &data).unwrap();
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
fn test_tensor_matmul() {
    let data1d: Vec<_> = (0..3).map(|x| x as f32).collect();
    let data2d: Vec<_> = (0..6).map(|x| x as f32).collect();
    let data3d: Vec<_> = (0..24).map(|x| x as f32).collect();

    // 1D x 1D
    let t1 = Tensor::shaped(&[3], &data1d).unwrap();
    let t2 = Tensor::shaped(&[3], &data1d).unwrap();
    let out = t1.matmul(&t2).unwrap();
    assert_scalar_tensor(&out, 5.0);

    // 1D x 2D
    let t1 = Tensor::shaped(&[3], &data1d).unwrap();
    let t2 = Tensor::shaped(&[3, 2], &data2d).unwrap();
    let out = t1.matmul(&t2).unwrap();
    assert_contiguous_tensor(&out, &[10.0, 13.0], &[2]);

    // 1D x 3D
    let t1 = Tensor::shaped(&[3], &data1d).unwrap();
    let t2 = Tensor::shaped(&[4, 3, 2], &data3d).unwrap();
    let out = t1.matmul(&t2).unwrap();
    assert_contiguous_tensor(
        &out,
        &[10.0, 13.0, 28.0, 31.0, 46.0, 49.0, 64.0, 67.0],
        &[4, 2],
    );

    // 2D x 1D
    let t1 = Tensor::shaped(&[2, 3], &data2d).unwrap();
    let t2 = Tensor::shaped(&[3], &data1d).unwrap();
    let out = t1.matmul(&t2).unwrap();
    assert_contiguous_tensor(&out, &[5.0, 14.0], &[2]);

    // 2D x 2D
    let t1 = Tensor::shaped(&[2, 3], &data2d).unwrap();
    let t2 = Tensor::shaped(&[3, 2], &data2d).unwrap();
    let out = t1.matmul(&t2).unwrap();
    assert_contiguous_tensor(&out, &[10.0, 13.0, 28.0, 40.0], &[2, 2]);

    // 2D x 3D
    let t1 = Tensor::shaped(&[2, 3], &data2d).unwrap();
    let t2 = Tensor::shaped(&[4, 3, 2], &data3d).unwrap();
    let out = t1.matmul(&t2).unwrap();
    assert_contiguous_tensor(
        &out,
        &[
            10.0, 13.0, 28.0, 40.0, 28.0, 31.0, 100.0, 112.0, 46.0, 49.0, 172.0, 184.0, 64.0, 67.0,
            244.0, 256.0,
        ],
        &[4, 2, 2],
    );

    // 3D x 1D
    let t1 = Tensor::shaped(&[4, 2, 3], &data3d).unwrap();
    let t2 = Tensor::shaped(&[3], &data1d).unwrap();
    let out = t1.matmul(&t2).unwrap();
    assert_contiguous_tensor(
        &out,
        &[5.0, 14.0, 23.0, 32.0, 41.0, 50.0, 59.0, 68.0],
        &[4, 2],
    );

    // 3D x 2D
    let t1 = Tensor::shaped(&[4, 2, 3], &data3d).unwrap();
    let t2 = Tensor::shaped(&[3, 2], &data2d).unwrap();
    let out = t1.matmul(&t2).unwrap();
    assert_contiguous_tensor(
        &out,
        &[
            10.0, 13.0, 28.0, 40.0, 46.0, 67.0, 64.0, 94.0, 82.0, 121.0, 100.0, 148.0, 118.0,
            175.0, 136.0, 202.0,
        ],
        &[4, 2, 2],
    );

    // 3D x 3D
    let t1 = Tensor::shaped(&[4, 2, 3], &data3d).unwrap();
    let t2 = Tensor::shaped(&[4, 3, 2], &data3d).unwrap();
    let out = t1.matmul(&t2).unwrap();
    assert_contiguous_tensor(
        &out,
        &[
            10.0, 13.0, 28.0, 40.0, 172.0, 193.0, 244.0, 274.0, 550.0, 589.0, 676.0, 724.0, 1144.0,
            1201.0, 1324.0, 1390.0,
        ],
        &[4, 2, 2],
    );

    // 3D x 3D
    let t1 = Tensor::shaped(&[1, 2, 3], &data2d).unwrap();
    let t2 = Tensor::shaped(&[4, 3, 2], &data3d).unwrap();
    let out = t1.matmul(&t2).unwrap();
    assert_contiguous_tensor(
        &out,
        &[
            10.0, 13.0, 28.0, 40.0, 28.0, 31.0, 100.0, 112.0, 46.0, 49.0, 172.0, 184.0, 64.0, 67.0,
            244.0, 256.0,
        ],
        &[4, 2, 2],
    );

    // 3D x 3D
    let t1 = Tensor::shaped(&[4, 2, 3], &data3d).unwrap();
    let t2 = Tensor::shaped(&[1, 3, 2], &data2d).unwrap();
    let out = t1.matmul(&t2).unwrap();
    assert_contiguous_tensor(
        &out,
        &[
            10.0, 13.0, 28.0, 40.0, 46.0, 67.0, 64.0, 94.0, 82.0, 121.0, 100.0, 148.0, 118.0,
            175.0, 136.0, 202.0,
        ],
        &[4, 2, 2],
    );
}

#[test]
fn test_tensor_into_iter() {
    let data = [2f32, 3f32, 4f32];
    let t = Tensor::from(data);
    for (x, y) in t.into_iter().zip(&data) {
        assert_eq!(x, *y);
    }

    let data = (0..24).map(|x| x as f32).collect::<Vec<_>>();
    let t = Tensor::shaped(&[2, 3, 4], &data).unwrap();
    for (x, y) in t.into_iter().zip(&data) {
        assert_eq!(x, *y);
    }
}
