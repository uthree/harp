pub fn setup() {
    // Ignore the result because the logger might have been initialized already in another test.
    let _ = env_logger::try_init();
}
