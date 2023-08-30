use candle::tensor::{layout::TensorLayout, Tensor};

pub fn assert_contiguous_layout(layout: &TensorLayout, shape: &[usize]) {
    let expected = TensorLayout::from(shape);
    assert_eq!(*layout, expected);
}

pub fn assert_reduced_layout(
    reduced: &TensorLayout,
    reducer: &TensorLayout,
    axis: &[usize],
    shape: &[usize],
) {
    assert_contiguous_layout(reduced, shape);
    let mut strides = reduced.strides().to_vec();
    for d in axis {
        strides[*d] = 0;
    }
    assert_eq!(reducer.shape(), shape);
    assert_eq!(reducer.strides(), strides);
}

pub fn assert_expanded_layout(expanded: &TensorLayout, original: &TensorLayout, shape: &[usize]) {
    let mut strides = Vec::with_capacity(shape.len());
    for ((d1, d2), stride) in original
        .shape()
        .iter()
        .rev()
        .zip(shape.iter().rev())
        .zip(original.strides().iter().rev())
    {
        if d1 == d2 {
            strides.push(*stride);
        } else {
            strides.push(0);
        }
    }
    while shape.len() > strides.len() {
        strides.push(0);
    }
    strides.reverse();
    assert_eq!(expanded.shape(), shape);
    assert_eq!(expanded.strides(), strides);
}

pub fn assert_contiguous_tensor(tensor: &Tensor, data: &[f32], shape: &[usize]) {
    assert_contiguous_layout(tensor.layout(), shape);
    let data_collected: Vec<_> = tensor.into_iter().copied().collect();
    assert_eq!(data_collected, data);
}

pub fn assert_zipped_tensor<F>(zipped: &Tensor, t1: &Tensor, t2: &Tensor, op: F)
where
    F: Fn(&f32, &f32) -> f32,
{
    let (l1, l2) = t1.layout().broadcast(t2.layout()).unwrap();
    for index in zipped.layout().iter_index() {
        let exp = op(
            &t1[l1.index_to_position(&index)],
            &t2[l2.index_to_position(&index)],
        );
        assert_eq!(zipped[index], exp);
    }
}
