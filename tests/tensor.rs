use candle::tensor::{ops::ToCpu, shape, Cpu, Tensor};

#[test]
fn i32_tensor() {
    let c = Cpu::<i32>::scalar(2);
    let mut t = Cpu::<i32>::new(&shape([2, 3]), &[1, 2, 3, 4, 5, 6]);
    t = &t + &t;
    t = &c * &t;
    assert_eq!(t.shape(), &shape([2, 3]));
    assert_eq!(t.ravel(), &[4, 8, 12, 16, 20, 24]);
}

#[test]
fn bool_tensor() {
    let t = Cpu::<bool>::new(&shape([2, 3]), &[true, true, false, false, true, true]);
    let r = &t.eq_elements(&t);
    assert_eq!(r.shape(), &shape([2, 3]));
    assert_eq!(r.ravel(), &[true, true, true, true, true, true]);
}

#[test]
fn arithmetics() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(t0: &Tensor<T, f32, I>, t1: &Tensor<T, f32, I>) {
        let r1 = t0.exp();
        assert_floats_eq(
            &r1.ravel(),
            &[
                2.718_281_7,
                7.389_056,
                20.085_537,
                54.59815,
                148.41316,
                403.4288,
            ],
        );

        let r2 = t0.ln();
        assert_floats_eq(
            &r2.ravel(),
            &[
                0.0,
                0.693_147_24,
                1.098_612_4,
                1.386_294_5,
                1.609_438_1,
                1.791_759_6,
            ],
        );

        let r3 = t0 + t1;
        assert_eq!(r3.ravel(), vec![7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);

        let r4 = t0 - t1;
        assert_eq!(r4.ravel(), vec![-5.0, -5.0, -5.0, -5.0, -5.0, -5.0]);

        let r5 = t0 / t1;
        assert_floats_eq(
            &r5.ravel(),
            &[
                0.166_666_67,
                0.285_714_3,
                0.375,
                0.444_444_45,
                0.5,
                0.545_454_56,
            ],
        );

        let r6 = t0 * t1;
        assert_eq!(r6.ravel(), vec![6.0, 14.0, 24.0, 36.0, 50.0, 66.0]);

        let r7 = t0.eq(t1);
        assert_eq!(r7.ravel(), vec![false; 6]);
    }

    let shape = shape([2, 3]);
    let t0 = Cpu::<f32>::new(&shape, &linspace(1., 6., 6));
    let t1 = Cpu::<f32>::new(&shape, &linspace(6., 11., 6));
    test(&t0, &t1);
}

#[test]
fn broadcasted_add() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(
        t0: &Tensor<T, f32, I>,
        t1: &Tensor<T, f32, I>,
        t2: &Tensor<T, f32, I>,
        t3: &Tensor<T, f32, I>,
    ) {
        let res = t0 + t1;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., 1., 2., 4., 5., 6.]);

        let res = t1 + t0;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., 1., 2., 4., 5., 6.]);

        let res = t0 + t2;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., 2., 4., 3., 5., 7.]);

        let res = t2 + t0;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., 2., 4., 3., 5., 7.]);

        let res = t0 + t3;
        assert_eq!(res.shape(), &shape([3, 2, 3]));
        assert_floats_eq(
            &res.ravel(),
            &[
                0., 1., 2., 4., 5., 6., 2., 3., 4., 6., 7., 8., 4., 5., 6., 8., 9., 10.,
            ],
        );

        let res = t3 + t0;
        assert_eq!(res.shape(), &shape([3, 2, 3]));
        assert_floats_eq(
            &res.ravel(),
            &[
                0., 1., 2., 4., 5., 6., 2., 3., 4., 6., 7., 8., 4., 5., 6., 8., 9., 10.,
            ],
        );
    }

    let t0 = Cpu::<f32>::new(&shape([2, 3]), &linspace(0., 5., 6));
    let t1 = Cpu::<f32>::new(&shape([2, 1]), &linspace(0., 1., 2));
    let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(0., 2., 3));
    let t3 = Cpu::<f32>::new(&shape([3, 2, 1]), &linspace(0., 5., 6));
    test(&t0, &t1, &t2, &t3);
}

#[test]
fn broadcasted_sub() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(
        t0: &Tensor<T, f32, I>,
        t1: &Tensor<T, f32, I>,
        t2: &Tensor<T, f32, I>,
        t3: &Tensor<T, f32, I>,
    ) {
        let res = t0 - t1;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., 1., 2., 2., 3., 4.]);

        let res = t1 - t0;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., -1., -2., -2., -3., -4.]);

        let res = t0 - t2;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., 0., 0., 3., 3., 3.]);

        let res = t2 - t0;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., 0., 0., -3., -3., -3.]);

        let res = t0 - t3;
        assert_eq!(res.shape(), &shape([3, 2, 3]));
        assert_floats_eq(
            &res.ravel(),
            &[
                0., 1., 2., 2., 3., 4., -2., -1., 0., 0., 1., 2., -4., -3., -2., -2., -1., 0.,
            ],
        );

        let res = t3 - t0;
        assert_eq!(res.shape(), &shape([3, 2, 3]));
        assert_floats_eq(
            &res.ravel(),
            &[
                0., -1., -2., -2., -3., -4., 2., 1., 0., 0., -1., -2., 4., 3., 2., 2., 1., 0.,
            ],
        );
    }

    let t0 = Cpu::<f32>::new(&shape([2, 3]), &linspace(0., 5., 6));
    let t1 = Cpu::<f32>::new(&shape([2, 1]), &linspace(0., 1., 2));
    let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(0., 2., 3));
    let t3 = Cpu::<f32>::new(&shape([3, 2, 1]), &linspace(0., 5., 6));
    test(&t0, &t1, &t2, &t3);
}

#[test]
fn broadcasted_mul() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(
        t0: &Tensor<T, f32, I>,
        t1: &Tensor<T, f32, I>,
        t2: &Tensor<T, f32, I>,
        t3: &Tensor<T, f32, I>,
    ) {
        let res = t0 * t1;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., 0., 0., 3., 4., 5.]);

        let res = t1 * t0;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., 0., 0., 3., 4., 5.]);

        let res = t0 * t2;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., 1., 4., 0., 4., 10.]);

        let res = t2 * t0;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[0., 1., 4., 0., 4., 10.]);

        let res = t0 * t3;
        assert_eq!(res.shape(), &shape([3, 2, 3]));
        assert_floats_eq(
            &res.ravel(),
            &[
                0., 0., 0., 3., 4., 5., 0., 2., 4., 9., 12., 15., 0., 4., 8., 15., 20., 25.,
            ],
        );

        let res = t3 * t0;
        assert_eq!(res.shape(), &shape([3, 2, 3]));
        assert_floats_eq(
            &res.ravel(),
            &[
                0., 0., 0., 3., 4., 5., 0., 2., 4., 9., 12., 15., 0., 4., 8., 15., 20., 25.,
            ],
        );
    }

    let t0 = Cpu::<f32>::new(&shape([2, 3]), &linspace(0., 5., 6));
    let t1 = Cpu::<f32>::new(&shape([2, 1]), &linspace(0., 1., 2));
    let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(0., 2., 3));
    let t3 = Cpu::<f32>::new(&shape([3, 2, 1]), &linspace(0., 5., 6));
    test(&t0, &t1, &t2, &t3);
}

#[test]
fn broadcasted_div() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(
        t0: &Tensor<T, f32, I>,
        t1: &Tensor<T, f32, I>,
        t2: &Tensor<T, f32, I>,
        t3: &Tensor<T, f32, I>,
    ) {
        let res = t0 / t1;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[1., 2., 3., 2., 2.5, 3.]);

        let res = t1 / t0;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(
            &res.ravel(),
            &[1., 0.5, 0.333_333_34, 0.5, 0.4, 0.333_333_34],
        );

        let res = t0 / t2;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[1., 1., 1., 4., 2.5, 2.]);

        let res = t2 / t0;
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[1., 1., 1., 0.25, 0.4, 0.5]);

        let res = t0 / t3;
        assert_eq!(res.shape(), &shape([3, 2, 3]));
        assert_floats_eq(
            &res.ravel(),
            &[
                1.,
                2.,
                3.,
                2.,
                2.5,
                3.,
                0.333_333_34,
                0.666_666_7,
                1.,
                1.,
                1.25,
                1.5,
                0.2,
                0.4,
                0.6,
                0.666_666_7,
                0.833_333_3,
                1.,
            ],
        );

        let res = t3 / t0;
        assert_eq!(res.shape(), &shape([3, 2, 3]));
        assert_floats_eq(
            &res.ravel(),
            &[
                1.,
                0.5,
                0.333_333_34,
                0.5,
                0.4,
                0.333_333_34,
                3.,
                1.5,
                1.,
                1.,
                0.8,
                0.666_666_7,
                5.,
                2.5,
                1.666_666_7,
                1.5,
                1.2,
                1.,
            ],
        );
    }

    let t0 = Cpu::<f32>::new(&shape([2, 3]), &linspace(1., 6., 6));
    let t1 = Cpu::<f32>::new(&shape([2, 1]), &linspace(1., 2., 2));
    let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(1., 3., 3));
    let t3 = Cpu::<f32>::new(&shape([3, 2, 1]), &linspace(1., 6., 6));
    test(&t0, &t1, &t2, &t3);
}

#[test]
fn broadcasted_pow() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(
        t0: &Tensor<T, f32, I>,
        t1: &Tensor<T, f32, I>,
        t2: &Tensor<T, f32, I>,
        t3: &Tensor<T, f32, I>,
    ) {
        let res = t0.pow(t1);
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[1., 1., 1., 3., 4., 5.]);

        let res = t1.pow(t0);
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[1., 0., 0., 1., 1., 1.]);

        let res = t0.pow(t2);
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[1., 1., 4., 1., 4., 25.]);

        let res = t2.pow(t0);
        assert_eq!(res.shape(), &shape([2, 3]));
        assert_floats_eq(&res.ravel(), &[1., 1., 4., 0., 1., 32.]);

        let res = t0.pow(t3);
        assert_eq!(res.shape(), &shape([3, 2, 3]));
        assert_floats_eq(
            &res.ravel(),
            &[
                1., 1., 1., 3., 4., 5., 0., 1., 4., 27., 64., 125., 0., 1., 16., 243., 1024., 3125.,
            ],
        );

        let res = t3.pow(t0);
        assert_eq!(res.shape(), &shape([3, 2, 3]));
        assert_floats_eq(
            &res.ravel(),
            &[
                1., 0., 0., 1., 1., 1., 1., 2., 4., 27., 81., 243., 1., 4., 16., 125., 625., 3125.,
            ],
        );
    }

    let t0 = Cpu::<f32>::new(&shape([2, 3]), &linspace(0., 5., 6));
    let t1 = Cpu::<f32>::new(&shape([2, 1]), &linspace(0., 1., 2));
    let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(0., 2., 3));
    let t3 = Cpu::<f32>::new(&shape([3, 2, 1]), &linspace(0., 5., 6));
    test(&t0, &t1, &t2, &t3);
}

#[test]
fn reduce_sum() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(t: &Tensor<T, f32, I>) {
        let t1 = t.sum(&[0]);
        assert_eq!(t1.shape(), &shape([1, 2, 2]));
        assert_floats_eq(&t1.ravel(), &[4., 6., 8., 10.]);

        let t2 = t.sum(&[1]);
        assert_eq!(t2.shape(), &shape([2, 1, 2]));
        assert_floats_eq(&t2.ravel(), &[2., 4., 10., 12.]);

        let t3 = t.sum(&[2]);
        assert_eq!(t3.shape(), &shape([2, 2, 1]));
        assert_floats_eq(&t3.ravel(), &[1., 5., 9., 13.]);

        let t4 = t.sum(&[0, 1]);
        assert_eq!(t4.shape(), &shape([1, 1, 2]));
        assert_floats_eq(&t4.ravel(), &[12., 16.]);

        let t5 = t.sum(&[0, 2]);
        assert_eq!(t5.shape(), &shape([1, 2, 1]));
        assert_floats_eq(&t5.ravel(), &[10., 18.]);

        let t6 = t.sum(&[1, 2]);
        assert_eq!(t6.shape(), &shape([2, 1, 1]));
        assert_floats_eq(&t6.ravel(), &[6., 22.]);

        let t7 = t.sum(&[0, 1, 2]);
        assert_eq!(t7.shape(), &shape([1, 1, 1]));
        assert_floats_eq(&t7.ravel(), &[28.]);
    }

    let t = Cpu::<f32>::new(&shape([2, 2, 2]), &linspace(0., 7., 8));
    test(&t);
}

#[test]
fn reduce_max() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(t: &Tensor<T, f32, I>) {
        let t1 = t.max(&[0]);
        assert_eq!(t1.shape(), &shape([1, 2, 2]));
        assert_floats_eq(&t1.ravel(), &[4., 5., 6., 7.]);

        let t2 = t.max(&[1]);
        assert_eq!(t2.shape(), &shape([2, 1, 2]));
        assert_floats_eq(&t2.ravel(), &[2., 3., 6., 7.]);

        let t3 = t.max(&[2]);
        assert_eq!(t3.shape(), &shape([2, 2, 1]));
        assert_floats_eq(&t3.ravel(), &[1., 3., 5., 7.]);

        let t4 = t.max(&[0, 1]);
        assert_eq!(t4.shape(), &shape([1, 1, 2]));
        assert_floats_eq(&t4.ravel(), &[6., 7.]);

        let t5 = t.max(&[0, 2]);
        assert_eq!(t5.shape(), &shape([1, 2, 1]));
        assert_floats_eq(&t5.ravel(), &[5., 7.]);

        let t6 = t.max(&[1, 2]);
        assert_eq!(t6.shape(), &shape([2, 1, 1]));
        assert_floats_eq(&t6.ravel(), &[3., 7.]);

        let t7 = t.max(&[0, 1, 2]);
        assert_eq!(t7.shape(), &shape([1, 1, 1]));
        assert_floats_eq(&t7.ravel(), &[7.]);
    }

    let t = Cpu::<f32>::new(&shape([2, 2, 2]), &linspace(0., 7., 8));
    test(&t);
}

#[test]
fn reshape() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(t: &Tensor<T, f32, I>) {
        let data = linspace(0., 23., 24);

        let t0 = t.reshape(&shape([6, 4]));
        assert_eq!(&t0.shape(), &shape([6, 4]));
        assert_floats_eq(&t0.ravel(), &data);

        let t1 = t.reshape(&shape([2, 12]));
        assert_eq!(&t1.shape(), &shape([2, 12]));
        assert_floats_eq(&t1.ravel(), &data);

        let t2 = t.reshape(&shape([1, 6, 4]));
        assert_eq!(&t2.shape(), &shape([1, 6, 4]));
        assert_floats_eq(&t2.ravel(), &data);

        let t3 = t.reshape(&shape([2, 6, 2]));
        assert_eq!(&t3.shape(), &shape([2, 6, 2]));
        assert_floats_eq(&t3.ravel(), &data);

        let t4 = t.reshape(&shape([2, 3, 2, 2]));
        assert_eq!(&t4.shape(), &shape([2, 3, 2, 2]));
        assert_floats_eq(&t4.ravel(), &data);

        let t5 = t.reshape(&shape([3, 8]));
        assert_eq!(&t5.shape(), &shape([3, 8]));
        assert_floats_eq(&t5.ravel(), &data);

        let t6 = t.reshape(&shape([8, 3]));
        assert_eq!(&t6.shape(), &shape([8, 3]));
        assert_floats_eq(&t6.ravel(), &data);
    }

    let t = Cpu::<f32>::new(&shape([2, 3, 4]), &linspace(0., 23., 24));
    test(&t);
}

#[test]
fn permute() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(t: &Tensor<T, f32, I>) {
        let t0 = t.permute(&[0, 1, 2]);
        assert_eq!(&t0.shape(), &shape([1, 2, 3]));
        assert_floats_eq(&t0.ravel(), &[0., 1., 2., 3., 4., 5.]);

        let t0 = t.permute(&[0, 2, 1]);
        assert_eq!(&t0.shape(), &shape([1, 3, 2]));
        assert_floats_eq(&t0.ravel(), &[0., 3., 1., 4., 2., 5.]);

        let t0 = t.permute(&[1, 0, 2]);
        assert_eq!(&t0.shape(), &shape([2, 1, 3]));
        assert_floats_eq(&t0.ravel(), &[0., 1., 2., 3., 4., 5.]);

        let t0 = t.permute(&[1, 2, 0]);
        assert_eq!(&t0.shape(), &shape([2, 3, 1]));
        assert_floats_eq(&t0.ravel(), &[0., 1., 2., 3., 4., 5.]);

        let t0 = t.permute(&[2, 0, 1]);
        assert_eq!(&t0.shape(), &shape([3, 1, 2]));
        assert_floats_eq(&t0.ravel(), &[0., 3., 1., 4., 2., 5.]);

        let t0 = t.permute(&[2, 1, 0]);
        assert_eq!(&t0.shape(), &shape([3, 2, 1]));
        assert_floats_eq(&t0.ravel(), &[0., 3., 1., 4., 2., 5.]);
    }

    let t = Cpu::<f32>::new(&shape([1, 2, 3]), &linspace(0., 5., 6));
    test(&t);
}

#[test]
fn expand() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(
        t0: &Tensor<T, f32, I>,
        t1: &Tensor<T, f32, I>,
        t2: &Tensor<T, f32, I>,
    ) {
        let r = t0.expand(&shape([3]));
        assert_eq!(&r.shape(), &shape([3]));
        assert_floats_eq(&r.ravel(), &[420.69, 420.69, 420.69]);

        let r = t1.expand(&shape([3, 2]));
        assert_eq!(&r.shape(), &shape([3, 2]));
        assert_floats_eq(&r.ravel(), &[420., 69., 420., 69., 420., 69.]);

        let r = t2.expand(&shape([2, 3]));
        assert_eq!(&r.shape(), &shape([2, 3]));
        assert_floats_eq(&r.ravel(), &[420., 420., 420., 69., 69., 69.]);
    }

    let t0 = Cpu::<f32>::new(&shape([1]), &[420.69]);
    let t1 = Cpu::<f32>::new(&shape([1, 2]), &[420., 69.]);
    let t2 = Cpu::<f32>::new(&shape([2, 1]), &[420., 69.]);
    test(&t0, &t1, &t2);
}

#[test]
fn transpose() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(t: &Tensor<T, f32, I>) {
        let t0 = t.transpose(0, 1);
        assert_eq!(&t0.shape(), &shape([2, 1, 3]));
        assert_floats_eq(&t0.ravel(), &[0., 1., 2., 3., 4., 5.]);

        let t0 = t.transpose(0, 2);
        assert_eq!(&t0.shape(), &shape([3, 2, 1]));
        assert_floats_eq(&t0.ravel(), &[0., 3., 1., 4., 2., 5.]);

        let t0 = t.transpose(1, 2);
        assert_eq!(&t0.shape(), &shape([1, 3, 2]));
        assert_floats_eq(&t0.ravel(), &[0., 3., 1., 4., 2., 5.]);
    }

    let t = Cpu::<f32>::new(&shape([1, 2, 3]), &linspace(0., 5., 6));
    test(&t);
}

#[test]
fn squeeze() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(
        t0: &Tensor<T, f32, I>,
        t1: &Tensor<T, f32, I>,
        t2: &Tensor<T, f32, I>,
        t3: &Tensor<T, f32, I>,
    ) {
        let r = t0.squeeze();
        assert_eq!(r.shape(), &shape([2, 3]));
        assert_floats_eq(&r.ravel(), &[0., 1., 2., 3., 4., 5.]);

        let r = t1.squeeze();
        assert_eq!(r.shape(), &shape([2, 3]));
        assert_floats_eq(&r.ravel(), &[0., 1., 2., 3., 4., 5.]);

        let r = t2.squeeze();
        assert_eq!(r.shape(), &shape([2, 3]));
        assert_floats_eq(&r.ravel(), &[0., 1., 2., 3., 4., 5.]);

        let r = t3.squeeze();
        assert_eq!(r.shape(), &shape([]));
        assert_floats_eq(&r.ravel(), &[420.69]);
    }

    let t0 = Cpu::<f32>::new(&shape([1, 2, 3]), &linspace(0., 5., 6));
    let t1 = Cpu::<f32>::new(&shape([2, 1, 3]), &linspace(0., 5., 6));
    let t2 = Cpu::<f32>::new(&shape([2, 3, 1]), &linspace(0., 5., 6));
    let t3 = Cpu::<f32>::new(&shape([1, 1]), &[420.69]);
    test(&t0, &t1, &t2, &t3);
}

#[test]
fn matmul() {
    fn test<T, I: ToCpu<Repr<f32> = T>>(
        t0: &Tensor<T, f32, I>,
        t1: &Tensor<T, f32, I>,
        t2: &Tensor<T, f32, I>,
        t3: &Tensor<T, f32, I>,
    ) {
        let r = t0.matmul(t0);
        assert_eq!(r.shape(), &[]);
        assert_floats_eq(&r.ravel(), &[5.]);

        let r = t0.matmul(t1);
        assert_eq!(r.shape(), &shape([1]));
        assert_floats_eq(&r.ravel(), &[5.]);

        let r = t0.matmul(t3);
        assert_eq!(r.shape(), &shape([1, 3]));
        assert_floats_eq(&r.ravel(), &[15., 18., 21.]);

        let r = t2.matmul(t0);
        assert_eq!(r.shape(), &shape([1]));
        assert_floats_eq(&r.ravel(), &[5.]);

        let r = t2.matmul(t1);
        assert_eq!(r.shape(), &shape([1, 1]));
        assert_floats_eq(&r.ravel(), &[5.]);

        let r = t2.matmul(t3);
        assert_eq!(r.shape(), &shape([1, 1, 3]));
        assert_floats_eq(&r.ravel(), &[15., 18., 21.]);

        let r = t3.matmul(t0);
        assert_eq!(r.shape(), &shape([1, 3]));
        assert_floats_eq(&r.ravel(), &[5., 14., 23.]);

        let r = t3.matmul(t1);
        assert_eq!(r.shape(), &shape([1, 3, 1]));
        assert_floats_eq(&r.ravel(), &[5., 14., 23.]);

        let r = t3.matmul(t3);
        assert_eq!(r.shape(), &shape([1, 3, 3]));
        assert_floats_eq(&r.ravel(), &[15., 18., 21., 42., 54., 66., 69., 90., 111.]);
    }

    let t0 = Cpu::<f32>::new(&shape([3]), &linspace(0., 2., 3));
    let t1 = Cpu::<f32>::new(&shape([3, 1]), &linspace(0., 2., 3));
    let t2 = Cpu::<f32>::new(&shape([1, 3]), &linspace(0., 2., 3));
    let t3 = Cpu::<f32>::new(&shape([1, 3, 3]), &linspace(0., 8., 9));
    test(&t0, &t1, &t2, &t3);
}

#[allow(clippy::similar_names)]
fn linspace(start: f32, stop: f32, num: u16) -> Vec<f32> {
    let step = if num > 1 {
        (stop - start) / f32::from(num - 1)
    } else {
        0.0
    };
    let mut data = Vec::with_capacity(num.into());
    let mut point = start;
    for _i in 0..num {
        data.push(point);
        point += step;
    }
    data
}

fn assert_floats_eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter()
            .zip(b.iter())
            .all(|(a, b)| (a.is_nan() && b.is_nan()) || ((a - b).abs() <= f32::EPSILON)),
        "{a:?} != {b:?}"
    );
}
