//! 最適化中のログをキャプチャするモジュール

use log::{Metadata, Record};
use std::sync::{Arc, Mutex};

/// グローバルなログバッファ（スレッド間で共有）
static LOG_BUFFER: Mutex<Vec<String>> = Mutex::new(Vec::new());

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
    use env_logger::{Builder, Env};
    use std::io::Write;

    // RUST_LOGが設定されていない場合でも、デフォルトで"debug"レベル以上を出力
    let env_logger = Builder::from_env(Env::default().default_filter_or("debug"))
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

    let max_level = env_logger.filter(); // env_loggerのフィルタレベルを取得
    let capture_logger = CaptureLogger::new(Box::new(env_logger));

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
        if let Ok(mut buffer) = LOG_BUFFER.lock() {
            buffer.push(message);
        }
    }

    fn flush(&self) {
        if let Ok(inner) = self.inner.lock() {
            inner.flush();
        }
    }
}

/// ログキャプチャを開始
pub fn start_capture() {
    if let Ok(mut buffer) = LOG_BUFFER.lock() {
        buffer.clear();
    }
}

/// キャプチャされたログを取得
pub fn get_captured_logs() -> Vec<String> {
    LOG_BUFFER
        .lock()
        .map(|buffer| buffer.clone())
        .unwrap_or_default()
}

/// ログキャプチャをクリア
pub fn clear_logs() {
    if let Ok(mut buffer) = LOG_BUFFER.lock() {
        buffer.clear();
    }
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
        if let Ok(mut buffer) = LOG_BUFFER.lock() {
            buffer.push("Test log 1".to_string());
            buffer.push("Test log 2".to_string());
        }

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

    #[test]
    #[ignore] // グローバルロガーは一度しか設定できないため、並行テストでは動作しない
    fn test_real_log_capture() {
        // このテストは手動で実行する必要があります：
        // cargo test --lib opt::log_capture::tests::test_real_log_capture -- --ignored --test-threads=1

        // ロガーを初期化
        init_with_env_logger();

        // ログをクリア
        start_capture();

        // 実際のログマクロを使ってログを出力
        log::debug!("Debug message in test");
        log::info!("Info message in test");
        log::warn!("Warn message in test");

        // キャプチャされたログを取得
        let logs = get_captured_logs();

        // ログが記録されているはず
        assert!(
            !logs.is_empty(),
            "Logs should be captured. Got {} logs.",
            logs.len()
        );

        // ログの内容に期待する文字列が含まれているか確認
        let has_debug = logs.iter().any(|l| l.contains("Debug message in test"));
        let has_info = logs.iter().any(|l| l.contains("Info message in test"));
        let has_warn = logs.iter().any(|l| l.contains("Warn message in test"));

        if !has_debug || !has_info || !has_warn {
            eprintln!("Captured logs:");
            for log in &logs {
                eprintln!("  {}", log);
            }
        }

        assert!(has_debug, "Debug log should be captured. Logs: {:?}", logs);
        assert!(has_info, "Info log should be captured. Logs: {:?}", logs);
        assert!(has_warn, "Warn log should be captured. Logs: {:?}", logs);
    }
}
