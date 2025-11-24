//! harp-derive
//!
//! Module traitのderive macro実装

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// Module traitを自動実装するderive macro
///
/// # 使用例
///
/// ```ignore
/// use harp::prelude::*;
///
/// #[derive(Module)]
/// struct Linear {
///     weight: Parameter,
///     bias: Parameter,
/// }
/// ```
///
/// # フィールドの自動検出
///
/// - `Parameter`型のフィールド → パラメータとして登録
/// - それ以外の型のフィールド → サブモジュールとして登録（Module traitを実装している必要がある）
#[proc_macro_derive(Module, attributes(module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // structのフィールドを取得
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return syn::Error::new_spanned(
                    &input,
                    "Module derive only supports structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(&input, "Module derive only supports structs")
                .to_compile_error()
                .into();
        }
    };

    // Parameterフィールドとサブモジュールフィールドを分類
    let mut parameter_fields = Vec::new();
    let mut module_fields = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_type = &field.ty;

        // 型を文字列として取得
        let type_string = quote!(#field_type).to_string();

        // Parameter型かどうかを判定
        if type_string.contains("Parameter") {
            parameter_fields.push(field_name);
        } else {
            // Parameter型でない場合は、潜在的なサブモジュールとして扱う
            // （Module traitを実装しているかはコンパイル時にチェックされる）
            module_fields.push(field_name);
        }
    }

    // named_parameters()の実装を生成
    let named_params_impl = if parameter_fields.is_empty() && module_fields.is_empty() {
        quote! {
            std::collections::HashMap::new()
        }
    } else {
        let param_inserts = parameter_fields.iter().map(|field_name| {
            let field_name_str = field_name.to_string();
            quote! {
                params.insert(#field_name_str.to_string(), &self.#field_name);
            }
        });

        let module_inserts = module_fields.iter().map(|field_name| {
            let field_name_str = field_name.to_string();
            quote! {
                for (name, param) in self.#field_name.named_parameters() {
                    params.insert(format!("{}.{}", #field_name_str, name), param);
                }
            }
        });

        quote! {
            let mut params = std::collections::HashMap::new();
            #(#param_inserts)*
            #(#module_inserts)*
            params
        }
    };

    // named_parameters_mut()の実装を生成
    let named_params_mut_impl = if parameter_fields.is_empty() && module_fields.is_empty() {
        quote! {
            std::collections::HashMap::new()
        }
    } else {
        let param_inserts = parameter_fields.iter().map(|field_name| {
            let field_name_str = field_name.to_string();
            quote! {
                params.insert(#field_name_str.to_string(), &mut self.#field_name);
            }
        });

        let module_inserts = module_fields.iter().map(|field_name| {
            let field_name_str = field_name.to_string();
            quote! {
                for (name, param) in self.#field_name.named_parameters_mut() {
                    params.insert(format!("{}.{}", #field_name_str, name), param);
                }
            }
        });

        quote! {
            let mut params = std::collections::HashMap::new();
            #(#param_inserts)*
            #(#module_inserts)*
            params
        }
    };

    // Module trait実装を生成
    let expanded = quote! {
        impl #impl_generics harp::nn::Module for #name #ty_generics #where_clause {
            fn named_parameters(&self) -> std::collections::HashMap<String, &harp::nn::Parameter> {
                #named_params_impl
            }

            fn named_parameters_mut(&mut self) -> std::collections::HashMap<String, &mut harp::nn::Parameter> {
                #named_params_mut_impl
            }
        }
    };

    TokenStream::from(expanded)
}
