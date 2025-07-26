use crate::backends::{Backend, ClangBackend};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local! {
    /// A thread-local registry for backend instances.
    ///
    /// This ensures that for a given thread, all requests for a backend with the
    /// same name return the same instance, wrapped in an `Rc`. This is crucial
    /// for operations that involve multiple tensors, as they must share the same
    /// backend instance.
    static BACKEND_REGISTRY: RefCell<HashMap<String, Rc<dyn Backend>>> = RefCell::new(HashMap::new());
}

/// Retrieves a backend instance by name for the current thread.
///
/// If a backend of the given name has already been created on the current thread,
/// a reference to the existing instance is returned. Otherwise, a new instance
/// is created, stored in the thread-local registry, and a reference to it
/// is returned.
///
/// # Panics
///
/// Panics if the requested backend name is not supported.
///
/// # Example
///
/// ```
/// use harp::context::backend;
///
/// // Get the "clang" backend. This will create it if it doesn't exist.
/// let backend1 = backend("clang");
/// // Get it again. This will return the same instance.
/// let backend2 = backend("clang");
///
/// // Both Rc pointers point to the same allocation.
/// assert!(std::rc::Rc::ptr_eq(&backend1, &backend2));
/// ```
pub fn backend(name: &str) -> Rc<dyn Backend> {
    BACKEND_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        registry
            .entry(name.to_string())
            .or_insert_with(|| match name {
                "clang" => Rc::new(ClangBackend::new().unwrap()),
                _ => panic!("Unsupported backend: {name}"),
            })
            .clone()
    })
}
