pub mod assert;

use assert::{assert_contiguous_layout, assert_reduced_layout};
use candle::tensor::Layout;

use crate::assert::assert_expanded_layout;

#[test]
fn test_layout_reduce() {
    let layout = Layout::from(&[2, 3, 4]);
    let (reduced, reducer) = layout.reduce(&[0]).unwrap();
    assert_reduced_layout(&reduced, &reducer, &[0], &[1, 3, 4]);

    let (reduced, reducer) = layout.reduce(&[1]).unwrap();
    assert_reduced_layout(&reduced, &reducer, &[1], &[2, 1, 4]);

    let (reduced, reducer) = layout.reduce(&[2]).unwrap();
    assert_reduced_layout(&reduced, &reducer, &[2], &[2, 3, 1]);

    let (reduced, reducer) = layout.reduce(&[0, 1]).unwrap();
    assert_reduced_layout(&reduced, &reducer, &[0, 1], &[1, 1, 4]);

    let (reduced, reducer) = layout.reduce(&[0, 2]).unwrap();
    assert_reduced_layout(&reduced, &reducer, &[0, 2], &[1, 3, 1]);

    let (reduced, reducer) = layout.reduce(&[1, 2]).unwrap();
    assert_reduced_layout(&reduced, &reducer, &[1, 2], &[2, 1, 1]);

    let (reduced, reducer) = layout.reduce(&[0, 1, 2]).unwrap();
    assert_reduced_layout(&reduced, &reducer, &[0, 1, 2], &[1, 1, 1]);
}

#[test]
fn test_layout_squeeze() {
    let layout = Layout::from(&[1, 2, 3]);
    let squeezed = layout.squeeze();
    assert_contiguous_layout(&squeezed, &[2, 3]);

    let layout = Layout::from(&[2, 1, 3]);
    let squeezed = layout.squeeze();
    assert_contiguous_layout(&squeezed, &[2, 3]);

    let layout = Layout::from(&[2, 3, 1]);
    let squeezed = layout.squeeze();
    assert_contiguous_layout(&squeezed, &[2, 3]);

    let layout = Layout::from(&[1, 1, 1]);
    let squeezed = layout.squeeze();
    assert_contiguous_layout(&squeezed, &[]);
}

#[test]
fn test_layout_transpose() {
    let layout = Layout::from(&[2, 3, 4]);

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
    let layout = Layout::from(&[2, 3, 4]);

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
    let layout = Layout::from(&[1]);
    let expanded = layout.expand(&[3]).unwrap();
    assert_expanded_layout(&expanded, &layout, &[3]);

    let layout = Layout::from(&[1]);
    let expanded = layout.expand(&[3, 2]).unwrap();
    assert_expanded_layout(&expanded, &layout, &[3, 2]);

    let layout = Layout::from(&[2, 1, 1]);
    let expanded = layout.expand(&[7, 2, 4, 5]).unwrap();
    assert_expanded_layout(&expanded, &layout, &[7, 2, 4, 5]);

    let layout = Layout::from(&[1, 1, 2]);
    let expanded = layout.expand(&[4, 3, 2]).unwrap();
    assert_expanded_layout(&expanded, &layout, &[4, 3, 2]);
}

#[test]
fn test_layout_broadcast() {
    let t1 = Layout::from(&[1]);
    let t2 = Layout::from(&[3]);
    let (r1, r2) = t1.broadcast(&t2).unwrap();
    assert_expanded_layout(&r1, &t1, &[3]);
    assert_expanded_layout(&r2, &t2, &[3]);

    let t1 = Layout::from(&[3]);
    let t2 = Layout::from(&[1]);
    let (r1, r2) = t1.broadcast(&t2).unwrap();
    assert_expanded_layout(&r1, &t1, &[3]);
    assert_expanded_layout(&r2, &t2, &[3]);

    let t1 = Layout::from(&[2, 3]);
    let t2 = Layout::from(&[1]);
    let (r1, r2) = t1.broadcast(&t2).unwrap();
    assert_expanded_layout(&r1, &t1, &[2, 3]);
    assert_expanded_layout(&r2, &t2, &[2, 3]);

    let t1 = Layout::from(&[3, 2]);
    let t2 = Layout::from(&[1]);
    let (r1, r2) = t1.broadcast(&t2).unwrap();
    assert_expanded_layout(&r1, &t1, &[3, 2]);
    assert_expanded_layout(&r2, &t2, &[3, 2]);

    let t1 = Layout::from(&[2, 1, 4]);
    let t2 = Layout::from(&[7, 2, 4, 1]);
    let (r1, r2) = t1.broadcast(&t2).unwrap();
    assert_expanded_layout(&r1, &t1, &[7, 2, 4, 4]);
    assert_expanded_layout(&r2, &t2, &[7, 2, 4, 4]);

    let t1 = Layout::from(&[1, 4, 1, 2]);
    let t2 = Layout::from(&[1, 3, 1]);
    let (r1, r2) = t1.broadcast(&t2).unwrap();
    assert_expanded_layout(&r1, &t1, &[1, 4, 3, 2]);
    assert_expanded_layout(&r2, &t2, &[1, 4, 3, 2]);
}

#[test]
fn test_layout_reshape() {
    let layout = Layout::from(&[2, 3, 4]);

    let l = layout.reshape(&[6, 4]).unwrap().unwrap();
    assert_contiguous_layout(&l, &[6, 4]);

    let l = layout.reshape(&[2, 12]).unwrap().unwrap();
    assert_contiguous_layout(&l, &[2, 12]);

    let l = layout.reshape(&[1, 6, 4]).unwrap().unwrap();
    assert_contiguous_layout(&l, &[1, 6, 4]);

    let l = layout.reshape(&[2, 6, 2]).unwrap().unwrap();
    assert_contiguous_layout(&l, &[2, 6, 2]);

    let l = layout.reshape(&[2, 3, 2, 2]).unwrap().unwrap();
    assert_contiguous_layout(&l, &[2, 3, 2, 2]);
}

#[test]
fn test_layout_translate() {
    let layout = Layout::from(&[2, 2, 2]);
    let indices: Vec<_> = (0..2)
        .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| vec![x, y, z])))
        .collect();
    for (exp, idx) in indices.iter().enumerate() {
        let pos = layout.translate(idx);
        assert_eq!(pos, exp);
    }
}

#[test]
fn test_layout_index_iterator() {
    let layout = Layout::from(&[2, 3, 4]);
    let indices: Vec<_> = (0..2)
        .flat_map(|x| (0..3).flat_map(move |y| (0..4).map(move |z| vec![x, y, z])))
        .collect();
    for (i, idx) in layout.iter().enumerate() {
        assert_eq!(idx, indices[i]);
    }
}
