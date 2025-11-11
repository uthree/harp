//! 最適化中のログをキャプチャするモジュール

use log::{Metadata, Record};
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::Mutex;

thread_local! {
    /// スレッドローカルなログバッファ
    static LOG_BUFFER: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// ログをキャプチャするLogger
pub struct CaptureLogger {
    inner: Arc<Mutex<Box<dyn log::Log>>>,
}

impl CaptureLogger {
    /// 新しいCaptureLoggerを作成
    pub fn new(inner: Box<dyn log::Log>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }
}

/// CaptureLoggerをenv_loggerでラップして初期化する
///
/// この関数は、env_loggerの機能を保持しつつ、ログをキャプチャできるようにします。
/// アプリケーションの初期化時に、`env_logger::init()`の代わりに呼び出してください。
///
/// # Example
/// ```no_run
/// use harp::opt::log_capture;
///
/// // env_logger::init()の代わりに
/// log_capture::init_with_env_logger();
///
/// // 通常通りログを使用
/// log::info!("Application started");
/// ```
pub fn init_with_env_logger() {
    use env_logger::Builder;
    use std::io::Write;

    let env_logger = Builder::from_default_env()
        .format(|buf, record| {
            writeln!(
                buf,
                "[{}] {} - {}",
                record.level(),
                record.target(),
                record.args()
            )
        })
        .build();

    let capture_logger = CaptureLogger::new(Box::new(env_logger));
    let max_level = log::LevelFilter::Debug; // デフォルトでDebugレベル

    if log::set_boxed_logger(Box::new(capture_logger)).is_ok() {
        log::set_max_level(max_level);
    }
}

impl log::Log for CaptureLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        if let Ok(inner) = self.inner.lock() {
            inner.enabled(metadata)
        } else {
            false
        }
    }

    fn log(&self, record: &Record) {
        // 元のロガーにもログを送る
        if let Ok(inner) = self.inner.lock() {
            inner.log(record);
        }

        // バッファにもログを追加
        let message = format!(
            "[{}] {} - {}",
            record.level(),
            record.target(),
            record.args()
        );
        LOG_BUFFER.with(|buffer| {
            buffer.borrow_mut().push(message);
        });
    }

    fn flush(&self) {
        if let Ok(inner) = self.inner.lock() {
            inner.flush();
        }
    }
}

/// ログキャプチャを開始
pub fn start_capture() {
    LOG_BUFFER.with(|buffer| {
        buffer.borrow_mut().clear();
    });
}

/// キャプチャされたログを取得
pub fn get_captured_logs() -> Vec<String> {
    LOG_BUFFER.with(|buffer| buffer.borrow().clone())
}

/// ログキャプチャをクリア
pub fn clear_logs() {
    LOG_BUFFER.with(|buffer| {
        buffer.borrow_mut().clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_buffer_operations() {
        // ログバッファの基本操作をテスト
        // （実際のログキャプチャは、オプティマイザのテストで間接的にテストされる）

        // ログをクリア
        start_capture();
        assert_eq!(get_captured_logs().len(), 0);

        // LOG_BUFFERに直接書き込んでテスト
        LOG_BUFFER.with(|buffer| {
            buffer.borrow_mut().push("Test log 1".to_string());
            buffer.borrow_mut().push("Test log 2".to_string());
        });

        // ログを取得
        let logs = get_captured_logs();
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0], "Test log 1");
        assert_eq!(logs[1], "Test log 2");

        // ログをクリア
        clear_logs();
        let logs_after_clear = get_captured_logs();
        assert_eq!(logs_after_clear.len(), 0, "Logs should be cleared");
    }
}
