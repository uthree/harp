use harp::node::{self, Node};
use harp::rewriter;
use log::{Level, Metadata, Record};
use std::sync::{Mutex, Once};

// A simple logger for testing purposes that captures logs.
struct TestLogger {
    logs: Mutex<Vec<String>>,
}

impl log::Log for TestLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Debug
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            self.logs.lock().unwrap().push(format!("{}", record.args()));
        }
    }

    fn flush(&self) {}
}

static LOGGER: TestLogger = TestLogger {
    logs: Mutex::new(Vec::new()),
};
static INIT: Once = Once::new();

fn setup_test_logger() {
    INIT.call_once(|| {
        log::set_logger(&LOGGER)
            .map(|()| log::set_max_level(log::LevelFilter::Debug))
            .unwrap();
    });
    // Clear previous logs
    LOGGER.logs.lock().unwrap().clear();
}

#[test]
fn test_rewriter_log_crate() {
    setup_test_logger();

    let rewriter = rewriter!([
        (
            let x = capture("x")
            => x + Node::from(0.0f32)
            => |x| Some(x)
        )
    ]);

    let graph = node::constant(10.0f32) + 0.0f32;
    let _ = rewriter.rewrite(graph);

    let logs = LOGGER.logs.lock().unwrap();
    assert_eq!(logs.len(), 1);
    assert!(logs[0].starts_with("[Rewrite]"));
    assert!(logs[0].contains("OpAdd"));
    assert!(logs[0].contains("->"));
    assert!(logs[0].contains("Const(10.0)"));
}
