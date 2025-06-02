extern crate proc_macro;
use std::iter::once;

use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, parse_quote, DeriveInput, TypePath, Variant,
};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug)]
enum TemplateToken {
    Template(String),
    Verbatim(String),
}

struct Buffer {
    buffer: Option<String>,
    template_boundary: bool,
}

fn tokenize(str: &str) -> impl Iterator<Item = TemplateToken> + '_ {
    let mut buf = Buffer {
        buffer: None,
        template_boundary: false,
    };
    str.split_word_bounds()
        .into_iter()
        .chain(once("%"))
        .filter_map(move |word| {
            if buf.template_boundary {
                buf.template_boundary = false;
                Some(TemplateToken::Template(word.to_string()))
            } else if word == "%" {
                buf.template_boundary = true;
                buf.buffer.take().map(TemplateToken::Verbatim)
            } else {
                let mut new_buf = buf.buffer.take().unwrap_or_default();
                new_buf.push_str(word);
                buf.buffer.replace(new_buf);
                None
            }
        })
}

#[derive(FromDeriveInput)]
#[darling(attributes(parse))]
struct ParsableOpts {
    #[darling(default)]
    pattern: String,
    token_type: Option<syn::Path>,
}

impl ParsableOpts {
    fn get_token_type(&self) -> syn::Type {
        let token_type = self
            .token_type
            .clone()
            .unwrap_or_else(|| parse_quote!(Token));
        syn::Type::Path(TypePath {
            qself: None,
            path: token_type,
        })
    }

    fn get_template(self) -> Vec<TemplateToken> {
        tokenize(&self.pattern).collect()
    }
}

struct Field<'a> {
    ty: &'a syn::Type,
    ident: syn::Ident,
}

fn get_fields(fields: &syn::Fields) -> Vec<Field<'_>> {
    match fields {
        syn::Fields::Named(fields) => fields
            .named
            .iter()
            .map(|f| Field {
                ty: &f.ty,
                ident: f.ident.clone().unwrap(),
            })
            .collect(),
        syn::Fields::Unnamed(fields) => fields
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, f)| Field {
                ty: &f.ty,
                ident: format_ident!("arg{i}"),
            })
            .collect(),
        syn::Fields::Unit => vec![],
    }
}

fn field_resolve_ref<'a>(fields: &'a Vec<Field<'a>>, ref_: &str) -> Option<&'a Field<'a>> {
    fields
        .iter()
        .find(|f| f.ident == ref_)
        .or_else(|| match ref_.parse::<usize>() {
            Ok(n) => fields.iter().nth(n),
            Err(_) => None,
        })
}

fn struct_make_constructor(ctor: proc_macro2::TokenStream, fields: &syn::Fields) -> proc_macro2::TokenStream {
    let field_data = get_fields(fields);
    let args = field_data
        .into_iter()
        .map(|Field { ident, .. }| quote!(#ident))
        .reduce(|acc, cur| quote!(#acc, #cur));
    match fields {
        syn::Fields::Named(_) => {
            quote! { #ctor { #args } }
        }
        syn::Fields::Unnamed(_) => {
            quote! { #ctor(#args) }
        }
        syn::Fields::Unit => quote!(#ctor),
    }
}

// #[proc_macro_attribute]
// pub fn todo(attr: TokenStream, item: TokenStream) -> TokenStream {
//     item
// }

fn derive_enum(opts: ParsableOpts, input: &syn::DeriveInput) -> TokenStream {
    let ident = &input.ident;
    let syn::Data::Enum(ref enum_) = input.data else {
        panic!("Not an enum")
    };

    let token_type = opts.get_token_type();

    let parser_code = enum_
        .variants
        .iter()
        .map(
            |&Variant {
                 ident: ref varident,
                 ref fields,
                 ..
             }| {
                let fields_data = get_fields(fields);
                let args = fields_data
                    .iter()
                    .map(|Field { ident, .. }| quote!(#ident))
                    .reduce(|acc, cur| quote!((#acc, #cur)))
                    .expect("Should be non empty");

                let parser_code = fields_data
                    .into_iter()
                    .map(|Field { ty, .. }| quote!(<#ty as ami::prelude::Parsable>::parser()))
                    .reduce(|acc, cur| quote!(#acc.then(#cur)))
                    .expect("Should be non empty");

                let ident = quote!(#ident::#varident);
                let constructor_code = struct_make_constructor(ident, fields);
                
                quote! {
                    #parser_code
                        .map(|#args| #constructor_code)
                        .boxed()
                }
            },
        )
        .reduce(|acc, cur| quote!(#acc, #cur));

    let full_code = quote! {
        impl Parsable for #ident {
            type Token = #token_type;
            fn parser() -> impl ami::core::Parser<Token = Self::Token, Expression = Self> {
                ami::parsers::one_of([
                    #parser_code
                ])
            }
        }
    };

    println!("{}", full_code.to_string());

    full_code.into()
}

fn struct_make_parser(
    fields: &Vec<Field<'_>>,
    template: &[TemplateToken],
) -> proc_macro2::TokenStream {
    template
        .iter()
        .map(|t| match t {
            TemplateToken::Template(template) => {
                let field = field_resolve_ref(fields, template).unwrap();
                let ty = &field.ty;
                quote!(<#ty as ami::prelude::Parsable>::parser())
            }
            TemplateToken::Verbatim(s) => quote!(ami::parsers::raw_sequence::<Token>(#s)),
        })
        .reduce(|acc, cur| quote!(#acc.then(#cur)))
        .expect("Should be non empty")
}

fn derive_struct(opts: ParsableOpts, input: &syn::DeriveInput) -> TokenStream {
    let ident = &input.ident;
    let syn::Data::Struct(ref struct_) = input.data else {
        panic!()
    };

    let token_type = opts.get_token_type();
    let template = opts.get_template();

    let fields = get_fields(&struct_.fields);
    let parser_code = struct_make_parser(&fields, &template[..]);
    let constructor_code = struct_make_constructor(quote!(#ident), &struct_.fields);

    let map_params_code = template
        .iter()
        .map(|t| {
            let ident = match t {
                TemplateToken::Template(template) => field_resolve_ref(&fields, &template)
                    .map(|f| f.ident.clone())
                    .expect(&format!("Unknown field {template}")),
                TemplateToken::Verbatim(_) => format_ident!("_"),
            };
            quote!(#ident)
        })
        .reduce(|acc, cur| quote!((#acc, #cur)))
        .unwrap_or_default();
    let map_code = quote! {
        let parser = parser.map(|#map_params_code| #constructor_code);
    };

    let full_code = quote! {
        impl Parsable for #ident {
            type Token = #token_type;
            fn parser() -> impl ami::core::Parser<Token = Self::Token, Expression = Self> {
                let parser = #parser_code;
                #map_code
                parser
            }
        }
    };

    println!("{}", full_code.to_string());

    full_code.into()
}

#[proc_macro_derive(Parsable, attributes(parse))]
pub fn derive_parsable(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let opts = ParsableOpts::from_derive_input(&input).unwrap();

    match input.data {
        syn::Data::Struct(_) => derive_struct(opts, &input),
        syn::Data::Enum(_) => derive_enum(opts, &input),
        _ => unimplemented!(),
    }
}
