use crate::backend::{
    c::{CBuffer, CCompiler, CRenderer},
    generic::GenericBackend,
};

pub type CBackend = GenericBackend<CCompiler, CRenderer, CBuffer>;
