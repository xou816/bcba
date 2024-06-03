use std::fmt::Display;

use ami::token::TokenProducer;

#[derive(Debug, Eq, Clone)]
pub enum Token {
    NameAnchor,
    LedgerEntryStart,
    DollarSymbol,
    Comment(String),
    KeywordPaid,
    KeywordFor,
    KeywordEveryone,
    KeywordBut,
    LineEnd,
    Comma,
    Word(String),
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::LedgerEntryStart => write!(f, "start of ledger"),
            Token::KeywordPaid => write!(f, "keyword `paid`"),
            Token::NameAnchor => write!(f, "name anchor `@`"),
            Token::Word(w) => write!(f, "token `{}`", w),
            Token::DollarSymbol => write!(f, "currency tag `$`"),
            Token::Comment(_) => write!(f, "comment"),
            Token::KeywordFor => write!(f, "keyword `for`"),
            Token::LineEnd => write!(f, "end of line"),
            Token::Comma => write!(f, "comma"),
            Token::KeywordEveryone => write!(f, "keyword `everyone`"),
            Token::KeywordBut => write!(f, "keyword `but`"),
        }
    }
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Word(l), Self::Word(r)) => l == r,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl TokenProducer for Token {
    type Token = Self;

    fn tokenize(word: &str, buffer: &mut ami::token::Buffer) -> Option<Self> {
        match word {
            "everyone" => Some(Token::KeywordEveryone),
            "but" => Some(Token::KeywordBut),
            "paid" => Some(Token::KeywordPaid),
            "for" => Some(Token::KeywordFor),
            "@" => Some(Token::NameAnchor),
            "-" => Some(Token::LedgerEntryStart),
            "$" => Some(Token::DollarSymbol),
            "(" => buffer.until(')'),
            ")" => buffer.done(|s| Token::Comment(s)),
            "\n" => Some(Token::LineEnd),
            "," => Some(Token::Comma),
            _ => Some(Token::Word(word.to_string())),
        }
    }
}