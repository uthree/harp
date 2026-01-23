//! UI drawing logic

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use eclat::backend::renderer::CLikeRenderer;

use super::app::App;

/// Main draw function
pub fn draw<R: CLikeRenderer + Clone>(frame: &mut Frame, app: &App<R>) {
    // Layout: top=main area, bottom=status bar
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(frame.area());

    // Main area: left=code, right=candidates list
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(chunks[0]);

    draw_code_panel(frame, app, main_chunks[0]);
    draw_candidates_panel(frame, app, main_chunks[1]);
    draw_status_bar(frame, app, chunks[1]);
}

/// Draw code panel
fn draw_code_panel<R: CLikeRenderer + Clone>(frame: &mut Frame, app: &App<R>, area: Rect) {
    let highlighted = app.highlight_current_code();

    // Create Line for each line
    let lines: Vec<Line> = highlighted
        .into_iter()
        .map(|line_spans| {
            let spans: Vec<Span> = line_spans
                .into_iter()
                .map(|(style, text)| Span::styled(text, style))
                .collect();
            Line::from(spans)
        })
        .collect();

    let text = Text::from(lines);

    let title = format!(
        " Code (Candidate {}/{}) ",
        app.selected_candidate() + 1,
        app.candidate_count()
    );

    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default()),
        )
        .scroll((app.scroll_offset(), 0));

    frame.render_widget(paragraph, area);
}

/// Draw candidates panel
fn draw_candidates_panel<R: CLikeRenderer + Clone>(frame: &mut Frame, app: &App<R>, area: Rect) {
    let snapshot = app.current_snapshot();

    let items: Vec<ListItem> = if let Some(snapshot) = snapshot {
        let mut items = Vec::new();

        // Selected candidate (rank 0)
        let selected_style = if app.selected_candidate() == 0 {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let selected_marker = if app.selected_candidate() == 0 {
            "> "
        } else {
            "  "
        };

        items.push(ListItem::new(Line::from(vec![
            Span::raw(selected_marker),
            Span::styled(
                format!(
                    "[0] cost={:.2} {}",
                    snapshot.cost,
                    snapshot.suggester_name.as_deref().unwrap_or("(initial)")
                ),
                selected_style,
            ),
        ])));

        // Alternative candidates
        for (i, alt) in snapshot.alternatives.iter().enumerate() {
            let alt_style = if app.selected_candidate() == i + 1 {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };

            let alt_marker = if app.selected_candidate() == i + 1 {
                "> "
            } else {
                "  "
            };

            items.push(ListItem::new(Line::from(vec![
                Span::raw(alt_marker),
                Span::styled(
                    format!(
                        "[{}] cost={:.2} {}",
                        i + 1,
                        alt.cost,
                        alt.suggester_name.as_deref().unwrap_or("(unknown)")
                    ),
                    alt_style,
                ),
            ])));
        }

        items
    } else {
        vec![ListItem::new(Span::raw("(No candidates)"))]
    };

    let list = List::new(items).block(
        Block::default()
            .title(" Candidates ")
            .borders(Borders::ALL)
            .border_style(Style::default()),
    );

    frame.render_widget(list, area);
}

/// Draw status bar
fn draw_status_bar<R: CLikeRenderer + Clone>(frame: &mut Frame, app: &App<R>, area: Rect) {
    let snapshot = app.current_snapshot();

    let step_info = format!("Step: {}/{}", app.current_step() + 1, app.total_steps());

    let cost_info = snapshot
        .map(|s| format!("Cost: {:.2}", s.cost))
        .unwrap_or_else(|| "Cost: N/A".to_string());

    let suggester_info = snapshot
        .and_then(|s| s.suggester_name.as_ref())
        .map(|name| format!("Suggester: {}", name))
        .unwrap_or_else(|| "Suggester: (none)".to_string());

    let description = snapshot.map(|s| s.description.clone()).unwrap_or_default();

    let status_text = format!(
        " {} | {} | {} | {} ",
        step_info, cost_info, suggester_info, description
    );

    let paragraph = Paragraph::new(status_text)
        .style(Style::default().fg(Color::White))
        .block(Block::default().borders(Borders::ALL));

    frame.render_widget(paragraph, area);
}
