extern crate proc_macro;
use std::iter::once;

use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, DeriveInput, Item};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug)]
enum MacroToken {
    Template(String),
    Verbatim(String),
}

struct Buffer {
    buffer: Option<String>,
    template_boundary: bool,
}

fn tokenize(str: &str) -> impl Iterator<Item = MacroToken> + '_ {
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
                Some(MacroToken::Template(word.to_string()))
            } else if word == "%" {
                buf.template_boundary = true;
                buf.buffer.take().map(MacroToken::Verbatim)
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
    pattern: Option<String>,
}

fn fields_resolve_ref<'a>(fields: &'a syn::Fields, ref_: &'a str) -> Option<&'a syn::Field> {
    match fields {
        syn::Fields::Named(fields) => fields
            .named
            .iter()
            .find(|f| f.ident.as_ref().unwrap() == ref_),
        syn::Fields::Unnamed(fields) => {
            let index = usize::from_str_radix(&ref_, 10).ok()?;
            fields.unnamed.iter().nth(index)
        }
        syn::Fields::Unit => None,
    }
}

fn fields_make_ident(fields: &syn::Fields, ref_: &str) -> Option<syn::Ident> {
    match fields {
        syn::Fields::Named(fields) => fields
            .named
            .iter()
            .filter_map(|f| f.ident.as_ref())
            .find(|&i| i == ref_)
            .cloned(),
        syn::Fields::Unnamed(fields) => {
            let index = usize::from_str_radix(&ref_, 10).ok()?;
            if index < fields.unnamed.len() {
                Some(format_ident!("arg{index}"))
            } else {
                None
            }
        }
        syn::Fields::Unit => None,
    }
}

fn struct_make_constructor(input: &syn::DeriveInput) -> proc_macro2::TokenStream {
    let ident = &input.ident;
    let syn::Data::Struct(ref struc) = input.data else {
        panic!()
    };
    match &struc.fields {
        syn::Fields::Named(fields) => {
            let fields = fields
                .named
                .iter()
                .filter_map(|f| {
                    let ident = f.ident.clone()?;
                    Some(quote!(#ident))
                })
                .reduce(|acc, cur| quote!(#acc, #cur))
                .unwrap_or_default();
            quote! { #ident { #fields } }
        }
        syn::Fields::Unnamed(fields) => {
            let fields = fields
                .unnamed
                .iter()
                .enumerate()
                .filter_map(|(i, _)| {
                    let ident = format_ident!("arg{i}");
                    Some(quote!(#ident))
                })
                .reduce(|acc, cur| quote!(#acc, #cur))
                .unwrap_or_default();
            quote! { #ident(#fields) }
        }
        syn::Fields::Unit => quote!(#ident),
    }
}

#[proc_macro_attribute]
pub fn token_enum(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let i2 = item.clone();
    let input = parse_macro_input!(i2 as Item);
    let Item::Mod(_) = input else {
        panic!()
    };
    item
}   

#[proc_macro_derive(Parsable, attributes(parse))]
pub fn derive_parsable(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let opts = ParsableOpts::from_derive_input(&input).unwrap();
    let ident = &input.ident;
    let syn::Data::Struct(ref struc) = input.data else {
        panic!()
    };

    let pattern = opts.pattern.unwrap_or_default();
    let tokens = tokenize(&pattern).collect::<Vec<_>>();

    let parser_code = tokens
        .iter()
        .map(|t| match t {
            MacroToken::Template(template) => {
                let field = fields_resolve_ref(&struc.fields, template).unwrap();
                let ty = &field.ty;
                quote!(<#ty as ami::prelude::Parsable>::parser())
            }
            MacroToken::Verbatim(s) => quote!(raw_sequence::<Token>(#s)),
        })
        .reduce(|acc, cur| quote!(#acc.then(#cur)));
    let parser_code = quote! {
        let parser = #parser_code;
    };

    let constructor_code = struct_make_constructor(&input);

    let map_params_code = tokens
        .iter()
        .map(|t| {
            let ident = match t {
                MacroToken::Template(template) => {
                    fields_make_ident(&struc.fields, template).unwrap()
                }
                MacroToken::Verbatim(_) => format_ident!("_"),
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
            type Token = Token;
            fn parser() -> impl Parser<Token = Self::Token, Expression = Self> {
                #parser_code
                #map_code
                parser
            }
        }
    };

    println!("{}", full_code.to_string());

    full_code.into()
}
