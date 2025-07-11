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

use harp::node::{capture, constant};
use harp::pattern::{RewriteRule, Rewriter};

#[test]
fn test_rewriter_log_crate() {
    let rule = harp::rewrite_rule!(let x = capture("x"); x.clone() + constant(0.0f32) => x);
    let rewriter = Rewriter::new("test_rewriter", vec![rule]);
    let graph = constant(1.0f32) + constant(0.0f32);
    let _ = rewriter.rewrite(graph);
}

