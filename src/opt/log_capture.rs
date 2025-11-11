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
