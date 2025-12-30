//! キーボードイベント処理

use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};

/// ユーザーアクション
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// 終了
    Quit,
    /// 次のステップへ
    NextStep,
    /// 前のステップへ
    PrevStep,
    /// 次の候補を選択
    NextCandidate,
    /// 前の候補を選択
    PrevCandidate,
}

/// イベントを処理してアクションを返す
///
/// 100msのタイムアウトでイベントをポーリング
pub fn handle_events() -> io::Result<Option<Action>> {
    if event::poll(Duration::from_millis(100))?
        && let Event::Key(key) = event::read()?
    {
        return Ok(handle_key_event(key));
    }
    Ok(None)
}

/// キーイベントをアクションに変換
fn handle_key_event(key: KeyEvent) -> Option<Action> {
    // Ctrl+C で終了
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Some(Action::Quit);
    }

    match key.code {
        // 終了
        KeyCode::Char('q') | KeyCode::Esc => Some(Action::Quit),

        // ステップ移動（左右キー / h,l）
        KeyCode::Left | KeyCode::Char('h') => Some(Action::PrevStep),
        KeyCode::Right | KeyCode::Char('l') => Some(Action::NextStep),

        // 候補選択（上下キー / j,k）
        KeyCode::Up | KeyCode::Char('k') => Some(Action::PrevCandidate),
        KeyCode::Down | KeyCode::Char('j') => Some(Action::NextCandidate),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_mapping_quit() {
        let key = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
        assert_eq!(handle_key_event(key), Some(Action::Quit));

        let key = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        assert_eq!(handle_key_event(key), Some(Action::Quit));
    }

    #[test]
    fn test_key_mapping_navigation() {
        let key = KeyEvent::new(KeyCode::Left, KeyModifiers::NONE);
        assert_eq!(handle_key_event(key), Some(Action::PrevStep));

        let key = KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE);
        assert_eq!(handle_key_event(key), Some(Action::PrevStep));

        let key = KeyEvent::new(KeyCode::Right, KeyModifiers::NONE);
        assert_eq!(handle_key_event(key), Some(Action::NextStep));

        let key = KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE);
        assert_eq!(handle_key_event(key), Some(Action::NextStep));
    }

    #[test]
    fn test_key_mapping_candidate_selection() {
        let key = KeyEvent::new(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(handle_key_event(key), Some(Action::PrevCandidate));

        let key = KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE);
        assert_eq!(handle_key_event(key), Some(Action::PrevCandidate));

        let key = KeyEvent::new(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(handle_key_event(key), Some(Action::NextCandidate));

        let key = KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE);
        assert_eq!(handle_key_event(key), Some(Action::NextCandidate));
    }

    #[test]
    fn test_ctrl_c_quit() {
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(handle_key_event(key), Some(Action::Quit));
    }

    #[test]
    fn test_unknown_key() {
        let key = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
        assert_eq!(handle_key_event(key), None);
    }
}
