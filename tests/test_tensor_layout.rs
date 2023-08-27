use candle::tensor::layout::{broadcast_shape, TensorLayout};

#[test]
fn test_layout_create() {
    let layout = TensorLayout::from(&[2, 3, 4]);
    assert_eq!(layout.elems(), 24);
    assert_eq!(layout.shape(), &[2, 3, 4]);
    assert_eq!(layout.strides(), &[12, 4, 1]);
}

#[test]
fn test_layout_squeeze() {
    let layout = TensorLayout::from(&[1, 2, 3]);
    let squeezed = layout.squeeze();
    assert_eq!(squeezed.shape(), &[2, 3]);
    assert_eq!(squeezed.strides(), &[3, 1]);

    let layout = TensorLayout::from(&[2, 1, 3]);
    let squeezed = layout.squeeze();
    assert_eq!(squeezed.shape(), &[2, 3]);
    assert_eq!(squeezed.strides(), &[3, 1]);

    let layout = TensorLayout::from(&[2, 3, 1]);
    let squeezed = layout.squeeze();
    assert_eq!(squeezed.shape(), &[2, 3]);
    assert_eq!(squeezed.strides(), &[3, 1]);
}

#[test]
fn test_layout_transpose() {
    let layout = TensorLayout::from(&[2, 3, 4]);

    let transposed = layout.transpose(0, 1).unwrap();
    assert_eq!(transposed.shape(), &[3, 2, 4]);
    assert_eq!(transposed.strides(), &[4, 12, 1]);

    let transposed = layout.transpose(1, 0).unwrap();
    assert_eq!(transposed.shape(), &[3, 2, 4]);
    assert_eq!(transposed.strides(), &[4, 12, 1]);

    let transposed = layout.transpose(0, 2).unwrap();
    assert_eq!(transposed.shape(), &[4, 3, 2]);
    assert_eq!(transposed.strides(), &[1, 4, 12]);

    let transposed = layout.transpose(2, 0).unwrap();
    assert_eq!(transposed.shape(), &[4, 3, 2]);
    assert_eq!(transposed.strides(), &[1, 4, 12]);

    let transposed = layout.transpose(1, 2).unwrap();
    assert_eq!(transposed.shape(), &[2, 4, 3]);
    assert_eq!(transposed.strides(), &[12, 1, 4]);

    let transposed = layout.transpose(2, 1).unwrap();
    assert_eq!(transposed.shape(), &[2, 4, 3]);
    assert_eq!(transposed.strides(), &[12, 1, 4]);
}

#[test]
fn test_layout_permute() {
    let layout = TensorLayout::from(&[2, 3, 4]);

    let permuted = layout.permute(&[0, 1, 2]).unwrap();
    assert_eq!(permuted.shape(), &[2, 3, 4]);
    assert_eq!(permuted.strides(), &[12, 4, 1]);

    let permuted = layout.permute(&[0, 2, 1]).unwrap();
    assert_eq!(permuted.shape(), &[2, 4, 3]);
    assert_eq!(permuted.strides(), &[12, 1, 4]);

    let permuted = layout.permute(&[1, 0, 2]).unwrap();
    assert_eq!(permuted.shape(), &[3, 2, 4]);
    assert_eq!(permuted.strides(), &[4, 12, 1]);

    let permuted = layout.permute(&[1, 2, 0]).unwrap();
    assert_eq!(permuted.shape(), &[3, 4, 2]);
    assert_eq!(permuted.strides(), &[4, 1, 12]);

    let permuted = layout.permute(&[2, 0, 1]).unwrap();
    assert_eq!(permuted.shape(), &[4, 2, 3]);
    assert_eq!(permuted.strides(), &[1, 12, 4]);

    let permuted = layout.permute(&[2, 1, 0]).unwrap();
    assert_eq!(permuted.shape(), &[4, 3, 2]);
    assert_eq!(permuted.strides(), &[1, 4, 12]);
}

#[test]
fn test_layout_expand() {
    let layout = TensorLayout::from(&[1]);
    let expanded = layout.expand(&[3]).unwrap();
    assert_eq!(expanded.shape(), &[3]);
    assert_eq!(expanded.strides(), &[0]);

    let layout = TensorLayout::from(&[1]);
    let expanded = layout.expand(&[3, 2]).unwrap();
    assert_eq!(expanded.shape(), &[3, 2]);
    assert_eq!(expanded.strides(), &[0, 0]);

    let layout = TensorLayout::from(&[2, 1, 1]);
    let expanded = layout.expand(&[7, 2, 4, 5]).unwrap();
    assert_eq!(expanded.shape(), &[7, 2, 4, 5]);
    assert_eq!(expanded.strides(), &[0, 1, 0, 0]);

    let layout = TensorLayout::from(&[1, 1, 2]);
    let expanded = layout.expand(&[4, 3, 2]).unwrap();
    assert_eq!(expanded.shape(), &[4, 3, 2]);
    assert_eq!(expanded.strides(), &[0, 0, 1]);
}

#[test]
fn test_layout_index_to_position() {
    let layout = TensorLayout::from(&[2, 2, 2]);
    let indices: Vec<_> = (0..2)
        .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| vec![x, y, z])))
        .collect();
    for (exp, idx) in indices.iter().enumerate() {
        let pos = layout.index_to_position(idx);
        assert_eq!(pos, exp);
    }
}

#[test]
fn test_layout_position_to_index() {
    let layout = TensorLayout::from(&[2, 2, 2]);
    let indices: Vec<_> = (0..2)
        .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| vec![x, y, z])))
        .collect();
    for (pos, exp) in indices.iter().enumerate() {
        let idx = layout.position_to_index(pos);
        assert_eq!(idx.as_slice(), exp.as_slice());
    }
}

#[test]
fn test_broadcast_shape() {
    let s = broadcast_shape(&[1], &[3]).unwrap();
    assert_eq!(s, &[3]);

    let s = broadcast_shape(&[3], &[1]).unwrap();
    assert_eq!(s, &[3]);

    let s = broadcast_shape(&[2, 3], &[1]).unwrap();
    assert_eq!(s, &[2, 3]);

    let s = broadcast_shape(&[1], &[3, 2]).unwrap();
    assert_eq!(s, &[3, 2]);

    let s = broadcast_shape(&[2, 1, 4], &[7, 2, 4, 1]).unwrap();
    assert_eq!(s, &[7, 2, 4, 4]);

    let s = broadcast_shape(&[1, 4, 1, 2], &[1, 3, 1]).unwrap();
    assert_eq!(s, &[1, 4, 3, 2]);
}

#[test]
fn test_layout_index_iterator() {
    let layout = TensorLayout::from(&[2, 3, 4]);
    let indices: Vec<_> = (0..2)
        .flat_map(|x| (0..3).flat_map(move |y| (0..4).map(move |z| vec![x, y, z])))
        .collect();
    for (i, idx) in layout.iter_index().enumerate() {
        assert_eq!(idx, indices[i]);
    }
}

#[test]
fn test_layout_position_iterator() {
    let layout = TensorLayout::from(&[2, 3, 4]);
    for (i, out) in layout.iter_position().enumerate() {
        assert_eq!(out, i);
    }
}
