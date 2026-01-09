//! 探索プログレス表示の抽象化
//!
//! 異なる探索アルゴリズム（ビームサーチ、貪欲法、幅優先探索など）で
//! 共通して使用できるプログレス表示機能を提供する。

use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

// ============================================================================
// Progress State
// ============================================================================

/// 探索の進捗状態
#[derive(Debug, Clone)]
pub struct ProgressState {
    /// 現在のステップ番号（0始まり）
    pub current_step: usize,
    /// 最大ステップ数
    pub max_steps: usize,
    /// 表示メッセージ
    pub message: String,
}

impl ProgressState {
    /// 新しい進捗状態を作成
    pub fn new(current_step: usize, max_steps: usize, message: impl Into<String>) -> Self {
        Self {
            current_step,
            max_steps,
            message: message.into(),
        }
    }
}

/// 探索完了時の情報
#[derive(Debug, Clone)]
pub struct FinishInfo {
    /// 経過時間
    pub elapsed: Duration,
    /// 実行したステップ数
    pub steps: usize,
    /// 最大ステップ数
    pub max_steps: usize,
    /// 早期収束したか（max_stepsに達する前に終了したか）
    pub converged: bool,
    /// タスク名（"AST optimization" など）
    pub task_name: String,
}

impl FinishInfo {
    /// 新しい完了情報を作成
    pub fn new(
        elapsed: Duration,
        steps: usize,
        max_steps: usize,
        task_name: impl Into<String>,
    ) -> Self {
        Self {
            elapsed,
            steps,
            max_steps,
            converged: steps < max_steps,
            task_name: task_name.into(),
        }
    }
}

// ============================================================================
// SearchProgress Trait
// ============================================================================

/// 探索プログレス表示のトレイト
///
/// 異なる探索アルゴリズムで共通して使用できる進捗表示インターフェース。
///
/// # Example
///
/// ```ignore
/// use eclat::opt::progress::{SearchProgress, IndicatifProgress};
///
/// let mut progress = IndicatifProgress::new();
/// progress.start(100, "Optimization");
///
/// for step in 0..100 {
///     progress.update(&ProgressState::new(step, 100, format!("step {}", step + 1)));
///     // ... do work ...
/// }
///
/// progress.finish(&FinishInfo::new(elapsed, 100, 100, "Optimization"));
/// ```
pub trait SearchProgress: Send {
    /// 探索開始時に呼ばれる
    ///
    /// # Arguments
    /// * `max_steps` - 最大ステップ数
    /// * `task_name` - タスク名（完了メッセージに使用）
    fn start(&mut self, max_steps: usize, task_name: &str);

    /// 各ステップで呼ばれる
    ///
    /// # Arguments
    /// * `state` - 現在の進捗状態
    fn update(&mut self, state: &ProgressState);

    /// 探索完了時に呼ばれる
    ///
    /// # Arguments
    /// * `info` - 完了情報
    fn finish(&mut self, info: &FinishInfo);

    /// 探索が中断された場合に呼ばれる
    fn abort(&mut self);
}

// ============================================================================
// IndicatifProgress
// ============================================================================

/// indicatifを使用したCargoスタイルのプログレス表示
///
/// 以下の形式で表示される：
/// ```text
///  Optimizing [======>                 ] 25/100 step 26
/// ```
///
/// 完了時：
/// ```text
///     Finished AST optimization in 1.23s (converged after 50 steps)
/// ```
pub struct IndicatifProgress {
    pb: Option<ProgressBar>,
    start_time: Option<Instant>,
    task_name: String,
}

impl IndicatifProgress {
    /// 新しいIndicatifProgressを作成
    pub fn new() -> Self {
        Self {
            pb: None,
            start_time: None,
            task_name: String::new(),
        }
    }
}

impl Default for IndicatifProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchProgress for IndicatifProgress {
    fn start(&mut self, max_steps: usize, task_name: &str) {
        self.task_name = task_name.to_string();
        self.start_time = Some(Instant::now());

        let pb = ProgressBar::new(max_steps as u64);
        pb.set_style(
            ProgressStyle::with_template("{prefix:>12.cyan.bold} [{bar:24}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=> "),
        );
        pb.set_prefix("Optimizing");
        self.pb = Some(pb);
    }

    fn update(&mut self, state: &ProgressState) {
        if let Some(ref pb) = self.pb {
            pb.set_message(state.message.clone());
            pb.set_position(state.current_step as u64);
        }
    }

    fn finish(&mut self, info: &FinishInfo) {
        if let Some(pb) = self.pb.take() {
            pb.finish_and_clear();

            let time_str = format_duration(info.elapsed);
            if info.converged {
                println!(
                    "{:>12} {} in {} (converged after {} steps)",
                    "\x1b[1;32mFinished\x1b[0m", info.task_name, time_str, info.steps
                );
            } else {
                println!(
                    "{:>12} {} in {} ({} steps)",
                    "\x1b[1;32mFinished\x1b[0m", info.task_name, time_str, info.steps
                );
            }
        }
    }

    fn abort(&mut self) {
        if let Some(pb) = self.pb.take() {
            pb.abandon();
        }
    }
}

// ============================================================================
// NoOpProgress
// ============================================================================

/// プログレス表示なし（テスト用・高速実行用）
///
/// 全てのメソッドが何も行わない空実装。
/// テスト時やプログレス表示が不要な場合に使用する。
#[derive(Debug, Clone, Copy, Default)]
pub struct NoOpProgress;

impl NoOpProgress {
    /// 新しいNoOpProgressを作成
    pub fn new() -> Self {
        Self
    }
}

impl SearchProgress for NoOpProgress {
    fn start(&mut self, _max_steps: usize, _task_name: &str) {}
    fn update(&mut self, _state: &ProgressState) {}
    fn finish(&mut self, _info: &FinishInfo) {}
    fn abort(&mut self) {}
}

// ============================================================================
// Helpers
// ============================================================================

/// Duration を読みやすい文字列に変換
fn format_duration(d: Duration) -> String {
    if d.as_secs() > 0 {
        format!("{:.2}s", d.as_secs_f64())
    } else {
        format!("{}ms", d.as_millis())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_state_new() {
        let state = ProgressState::new(5, 100, "step 6");
        assert_eq!(state.current_step, 5);
        assert_eq!(state.max_steps, 100);
        assert_eq!(state.message, "step 6");
    }

    #[test]
    fn test_finish_info_converged() {
        let info = FinishInfo::new(Duration::from_secs(1), 50, 100, "test");
        assert!(info.converged);
        assert_eq!(info.steps, 50);
        assert_eq!(info.max_steps, 100);
    }

    #[test]
    fn test_finish_info_not_converged() {
        let info = FinishInfo::new(Duration::from_secs(1), 100, 100, "test");
        assert!(!info.converged);
    }

    #[test]
    fn test_noop_progress() {
        let mut progress = NoOpProgress::new();
        progress.start(100, "test");
        progress.update(&ProgressState::new(0, 100, "step 1"));
        progress.finish(&FinishInfo::new(Duration::from_secs(1), 100, 100, "test"));
        // NoOpProgress は何も行わないので、パニックしなければOK
    }

    #[test]
    fn test_format_duration_seconds() {
        let d = Duration::from_secs_f64(1.234);
        assert_eq!(format_duration(d), "1.23s");
    }

    #[test]
    fn test_format_duration_milliseconds() {
        let d = Duration::from_millis(123);
        assert_eq!(format_duration(d), "123ms");
    }
}
