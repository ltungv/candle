pub mod autodiff;

#[derive(Debug, Clone)]
pub struct Sample<const M: usize, const N: usize> {
    pub input: [f64; M],
    pub output: [f64; N],
}
