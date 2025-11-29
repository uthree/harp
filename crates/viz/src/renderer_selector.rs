//! レンダラー選択機能
//!
//! 実行時にバックエンドレンダラーを切り替えるための機能を提供します。

use harp::ast::renderer::render_ast_with;
use harp::ast::AstNode;
use harp::backend::c::CRenderer;
use harp::backend::metal::MetalRenderer;
use harp::backend::opencl::OpenCLRenderer;

/// 利用可能なレンダラータイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RendererType {
    /// C言語レンダラー（シングルスレッド）
    #[default]
    C,
    /// OpenCLレンダラー（GPU）
    OpenCL,
    /// Metalレンダラー（Apple GPU）
    Metal,
}

impl RendererType {
    /// すべてのレンダラータイプを取得
    pub fn all() -> &'static [RendererType] {
        &[RendererType::C, RendererType::OpenCL, RendererType::Metal]
    }

    /// 表示名を取得
    pub fn display_name(&self) -> &'static str {
        match self {
            RendererType::C => "C (CPU)",
            RendererType::OpenCL => "OpenCL (GPU)",
            RendererType::Metal => "Metal (Apple GPU)",
        }
    }

    /// 短い名前を取得
    pub fn short_name(&self) -> &'static str {
        match self {
            RendererType::C => "C",
            RendererType::OpenCL => "OpenCL",
            RendererType::Metal => "Metal",
        }
    }
}

/// 指定されたRendererTypeでASTをレンダリング
pub fn render_with_type(ast: &AstNode, renderer_type: RendererType) -> String {
    match renderer_type {
        RendererType::C => {
            let renderer = CRenderer::new();
            render_ast_with(ast, &renderer)
        }
        RendererType::OpenCL => {
            let renderer = OpenCLRenderer::new();
            render_ast_with(ast, &renderer)
        }
        RendererType::Metal => {
            let renderer = MetalRenderer::new();
            render_ast_with(ast, &renderer)
        }
    }
}

/// レンダラー選択UIを描画
///
/// # Returns
/// レンダラーが変更された場合はtrue
pub fn renderer_selector_ui(ui: &mut egui::Ui, current: &mut RendererType) -> bool {
    let mut changed = false;

    egui::ComboBox::from_label("Renderer")
        .selected_text(current.display_name())
        .show_ui(ui, |ui| {
            for renderer_type in RendererType::all() {
                if ui
                    .selectable_value(current, *renderer_type, renderer_type.display_name())
                    .changed()
                {
                    changed = true;
                }
            }
        });

    changed
}
