use std::fmt::Display;

use ami::{token::TokenizerV3, tokenizers::*};

#[derive(Debug, Clone)]
pub enum Token {
    NameAnchor,
    LedgerEntryStart,
    DollarSymbol,
    Comment,
    KeywordPaid,
    KeywordFor,
    KeywordEveryone,
    KeywordBut,
    LineEnd,
    Comma,
    Word(String),
    Price(f64),
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::LedgerEntryStart => write!(f, "start of ledger"),
            Token::KeywordPaid => write!(f, "keyword `paid`"),
            Token::NameAnchor => write!(f, "name anchor `@`"),
            Token::Word(w) => write!(f, "token `{}`", w),
            Token::DollarSymbol => write!(f, "currency tag `$`"),
            Token::Comment => write!(f, "comment"),
            Token::KeywordFor => write!(f, "keyword `for`"),
            Token::LineEnd => write!(f, "end of line"),
            Token::Comma => write!(f, "comma"),
            Token::KeywordEveryone => write!(f, "keyword `everyone`"),
            Token::KeywordBut => write!(f, "keyword `but`"),
            Token::Price(_) => write!(f, "numeric value"),
        }
    }
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Word(l), Self::Word(r)) => l == r,
            (Self::Price(l), Self::Price(r)) => l == r,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Eq for Token {}

pub fn tokenizer() -> TokenizerV3<Token> {
    TokenizerV3::new(vec![
        keyword("everyone", Token::KeywordEveryone),
        keyword("but", Token::KeywordBut),
        keyword("paid", Token::KeywordPaid),
        keyword("for", Token::KeywordFor),
        keyword("@", Token::NameAnchor),
        keyword("-", Token::LedgerEntryStart),
        keyword("$", Token::DollarSymbol),
        keyword("\n", Token::LineEnd),
        keyword(",", Token::Comma),
        delimited("(", ")", |_| Token::Comment),
        identifier(Token::Word),
        numeric(|n| match n {
            ami::token::Numeric64::Int(i) => Token::Price(i as f64),
            ami::token::Numeric64::Float(f) => Token::Price(f),
        }),
        ignore_whitespace(),
    ])
}
