//! Keyboard event handling

use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};

/// User action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Quit
    Quit,
    /// Go to next step
    NextStep,
    /// Go to previous step
    PrevStep,
    /// Select next candidate
    NextCandidate,
    /// Select previous candidate
    PrevCandidate,
}

/// Handle events and return action
///
/// Polls events with 100ms timeout
pub fn handle_events() -> io::Result<Option<Action>> {
    if event::poll(Duration::from_millis(100))?
        && let Event::Key(key) = event::read()?
    {
        return Ok(handle_key_event(key));
    }
    Ok(None)
}

/// Convert key event to action
fn handle_key_event(key: KeyEvent) -> Option<Action> {
    // Ctrl+C to quit
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Some(Action::Quit);
    }

    match key.code {
        // Quit
        KeyCode::Char('q') | KeyCode::Esc => Some(Action::Quit),

        // Step navigation (left/right keys or h,l)
        KeyCode::Left | KeyCode::Char('h') => Some(Action::PrevStep),
        KeyCode::Right | KeyCode::Char('l') => Some(Action::NextStep),

        // Candidate selection (up/down keys or j,k)
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
