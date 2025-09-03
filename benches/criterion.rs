use std::hint::black_box;

use candle::tensor::{cpu, shape, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

type T32 = Tensor<cpu::Tensor<f32>, f32, cpu::TensorOps>;

fn matmul(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(12345u64);
    let mut group = c.benchmark_group("matmul");
    for size in [4, 8, 16, 32, 64, 96, 128] {
        let shape = shape([size, size]);
        let elems: Vec<_> = (&mut rng).random_iter().take(size * size).collect();
        let t1_cpu = T32::new(&shape, &elems);
        let t2_cpu = T32::new(&shape, &elems);
        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |b, _| {
            b.iter(|| (black_box(t1_cpu.matmul(&t2_cpu))))
        });
    }
    group.finish();
}

criterion_group!(bench_matmul, matmul);
criterion_main!(bench_matmul);
