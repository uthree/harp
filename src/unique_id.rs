use std::sync::atomic::{AtomicUsize, Ordering};

#[allow(dead_code)]
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn next_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::SeqCst)
}
