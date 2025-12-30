//! Module derive マクロ
//!
//! `#[derive(Module)]` を使用して、`Module` トレイトの
//! `parameters()` と `load_parameters()` を自動実装します。

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Ident, Meta, Type};

/// フィールドの種類
enum FieldKind {
    /// Parameter<T, D> 型 - パラメータとして直接追加
    Parameter,
    /// Module を実装する型 - 再帰的に探索
    Module,
    /// スキップする型 (PhantomData, _ で始まるフィールド)
    Skip,
}

/// フィールド情報
struct FieldInfo {
    name: Ident,
    kind: FieldKind,
}

/// 型からフィールドの種類を判定
fn classify_type(ty: &Type) -> FieldKind {
    let type_str = quote!(#ty).to_string();

    // Parameter<T> or Parameter<T, D> パターン
    if type_str.starts_with("Parameter") || type_str.contains("Parameter <") {
        return FieldKind::Parameter;
    }

    // PhantomData パターン
    if type_str.contains("PhantomData") {
        return FieldKind::Skip;
    }

    // それ以外は Module として扱う
    FieldKind::Module
}

/// フィールドを分類
fn classify_fields(fields: &Fields) -> Vec<FieldInfo> {
    let named_fields = match fields {
        Fields::Named(named) => &named.named,
        _ => panic!("Module can only be derived for structs with named fields"),
    };

    named_fields
        .iter()
        .filter_map(|field| {
            let name = field.ident.clone()?;

            // _ で始まるフィールドはスキップ
            if name.to_string().starts_with('_') {
                return Some(FieldInfo {
                    name,
                    kind: FieldKind::Skip,
                });
            }

            let kind = classify_type(&field.ty);
            Some(FieldInfo { name, kind })
        })
        .collect()
}

/// parameters() メソッドのボディを生成
fn generate_parameters(fields: &[FieldInfo], crate_path: &TokenStream2) -> TokenStream2 {
    let mut param_inserts = Vec::new();
    let mut module_inserts = Vec::new();

    for field in fields {
        let name = &field.name;
        let name_str = name.to_string();

        match field.kind {
            FieldKind::Parameter => {
                // Parameter<T, D> を &mut dyn ParameterMut<T> にキャスト
                param_inserts.push(quote! {
                    params.insert(
                        #name_str.to_string(),
                        &mut self.#name as &mut dyn #crate_path::ParameterMut<T>
                    );
                });
            }
            FieldKind::Module => {
                module_inserts.push(quote! {
                    for (sub_name, param) in self.#name.parameters() {
                        params.insert(format!("{}.{}", #name_str, sub_name), param);
                    }
                });
            }
            FieldKind::Skip => {}
        }
    }

    quote! {
        #(#param_inserts)*
        #(#module_inserts)*
    }
}

/// load_parameters() メソッドのボディを生成
fn generate_load_parameters(fields: &[FieldInfo], crate_path: &TokenStream2) -> TokenStream2 {
    let mut param_loads = Vec::new();
    let mut module_loads = Vec::new();

    for field in fields {
        let name = &field.name;
        let name_str = name.to_string();

        match field.kind {
            FieldKind::Parameter => {
                // Tensor<T, DimDyn> を受け取り、set_dyn で設定
                param_loads.push(quote! {
                    if let Some(tensor) = params.get(#name_str) {
                        #crate_path::ParameterMut::set_dyn(&mut self.#name, tensor.clone());
                    }
                });
            }
            FieldKind::Module => {
                module_loads.push(quote! {
                    let prefix = concat!(#name_str, ".");
                    let sub_params: std::collections::HashMap<String, harp::tensor::Tensor<T, harp::tensor::DimDyn>> = params
                        .iter()
                        .filter_map(|(k, v)| {
                            k.strip_prefix(prefix).map(|sub_key| (sub_key.to_string(), v.clone()))
                        })
                        .collect();
                    self.#name.load_parameters(sub_params);
                });
            }
            FieldKind::Skip => {}
        }
    }

    quote! {
        #(#param_loads)*
        #(#module_loads)*
    }
}

/// アトリビュートからクレートパスを取得
///
/// `#[module(crate = "path")]` から "path" を取得
/// 指定がなければデフォルトの "harp_nn" を返す
fn get_crate_path(attrs: &[syn::Attribute]) -> TokenStream2 {
    for attr in attrs {
        if attr.path().is_ident("module") {
            if let Meta::List(meta_list) = &attr.meta {
                let tokens = meta_list.tokens.clone();
                let tokens_str = tokens.to_string();

                // "crate = \"path\"" 形式をパース
                if let Some(path) = tokens_str
                    .strip_prefix("crate")
                    .and_then(|s| s.trim().strip_prefix("="))
                    .map(|s| s.trim().trim_matches('"'))
                {
                    if path == "crate" {
                        return quote!(crate);
                    } else {
                        let ident = Ident::new(path, Span::call_site());
                        return quote!(#ident);
                    }
                }
            }
        }
    }

    // デフォルト: harp_nn
    quote!(harp_nn)
}

/// Module derive マクロ
///
/// 構造体に対して `Module<T>` トレイトを自動実装します。
///
/// # フィールドの分類
///
/// - `Parameter<T, D>` 型: パラメータとして直接追加
/// - `_` で始まるフィールド: スキップ
/// - `PhantomData`: スキップ
/// - その他: `Module<T>` を実装していると仮定し、再帰的に探索
///
/// # アトリビュート
///
/// - `#[module(crate = "crate")]`: クレート内部で使用する場合に指定
/// - `#[module(crate = "my_crate")]`: 別名でインポートしている場合に指定
///
/// # Example
///
/// ```ignore
/// use harp_nn::{Module, Parameter, Linear};
///
/// #[derive(Module)]
/// struct MyModel<T: FloatDType> {
///     linear1: Linear<T>,
///     linear2: Linear<T>,
///     scale: Parameter<T, Dim1>,
///     _marker: PhantomData<T>,
/// }
///
/// // クレート内部で使用する場合
/// #[derive(Module)]
/// #[module(crate = "crate")]
/// struct InternalModel<T: FloatDType> {
///     layer: Linear<T>,
/// }
/// ```
#[proc_macro_derive(Module, attributes(module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let generics = &input.generics;

    // クレートパスを取得
    let crate_path = get_crate_path(&input.attrs);

    // ジェネリクスを解析
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // フィールドを取得・分類
    let fields = match &input.data {
        Data::Struct(data) => classify_fields(&data.fields),
        _ => panic!("Module can only be derived for structs"),
    };

    // メソッドボディを生成
    let parameters_body = generate_parameters(&fields, &crate_path);
    let load_parameters_body = generate_load_parameters(&fields, &crate_path);

    let expanded = quote! {
        impl #impl_generics #crate_path::Module<T> for #name #ty_generics #where_clause {
            fn parameters(&mut self) -> std::collections::HashMap<String, &mut dyn #crate_path::ParameterMut<T>> {
                let mut params = std::collections::HashMap::new();
                #parameters_body
                params
            }

            fn load_parameters(&mut self, params: std::collections::HashMap<String, harp::tensor::Tensor<T, harp::tensor::DimDyn>>) {
                #load_parameters_body
            }
        }
    };

    TokenStream::from(expanded)
}
