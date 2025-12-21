//! レンダラー選択機能
//!
//! 実行時にバックエンドレンダラーを切り替えるための機能を提供します。

use harp::ast::AstNode;
use harp::backend::c_like::CLikeRenderer;

#[cfg(feature = "metal")]
use harp::backend::metal::MetalRenderer;

#[cfg(feature = "opencl")]
use harp::backend::opencl::OpenCLRenderer;

/// 利用可能なレンダラータイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RendererType {
    /// C-likeレンダラー（フォールバック）
    #[default]
    CLike,
    /// OpenCLレンダラー（GPU）
    #[cfg(feature = "opencl")]
    OpenCL,
    /// Metalレンダラー（Apple GPU）
    #[cfg(feature = "metal")]
    Metal,
}

impl RendererType {
    /// すべてのレンダラータイプを取得
    pub fn all() -> Vec<RendererType> {
        #[allow(unused_mut)]
        let mut types = vec![RendererType::CLike];

        #[cfg(feature = "opencl")]
        {
            types.push(RendererType::OpenCL);
        }

        #[cfg(feature = "metal")]
        {
            types.push(RendererType::Metal);
        }

        types
    }

    /// 表示名を取得
    pub fn display_name(&self) -> &'static str {
        match self {
            RendererType::CLike => "C-like (Generic)",
            #[cfg(feature = "opencl")]
            RendererType::OpenCL => "OpenCL (GPU)",
            #[cfg(feature = "metal")]
            RendererType::Metal => "Metal (Apple GPU)",
        }
    }

    /// 短い名前を取得
    pub fn short_name(&self) -> &'static str {
        match self {
            RendererType::CLike => "C-like",
            #[cfg(feature = "opencl")]
            RendererType::OpenCL => "OpenCL",
            #[cfg(feature = "metal")]
            RendererType::Metal => "Metal",
        }
    }
}

/// 指定されたRendererTypeでASTをレンダリング
pub fn render_with_type(ast: &AstNode, renderer_type: RendererType) -> String {
    match renderer_type {
        RendererType::CLike => {
            let mut renderer = harp::backend::c_like::GenericRenderer::default();
            renderer.render_program_clike(ast)
        }
        #[cfg(feature = "opencl")]
        RendererType::OpenCL => {
            let mut renderer = OpenCLRenderer::default();
            renderer.render_program_clike(ast)
        }
        #[cfg(feature = "metal")]
        RendererType::Metal => {
            let mut renderer = MetalRenderer::default();
            renderer.render_program_clike(ast)
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
                    .selectable_value(current, renderer_type, renderer_type.display_name())
                    .changed()
                {
                    changed = true;
                }
            }
        });

    changed
}
