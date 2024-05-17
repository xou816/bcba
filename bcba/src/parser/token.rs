use std::fmt::Display;

use ami::token::TokenDeserialize;

#[derive(Debug, Eq, Clone)]
pub enum Token {
    NameAnchor,
    LedgerEntryStart,
    DollarSymbol,
    CommentStart,
    CommentEnd,
    KeywordPaid,
    KeywordFor,
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
            Token::CommentStart => write!(f, "start of comment `(`"),
            Token::CommentEnd => write!(f, "end of comment `)`"),
            Token::KeywordFor => write!(f, "keyword `for`"),
            Token::LineEnd => write!(f, "end of line"),
            Token::Comma => write!(f, "comma"),
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

impl TokenDeserialize for Token {
    fn try_from_char(c: char) -> Option<Token> {
        match c {
            '@' => Some(Token::NameAnchor),
            '-' => Some(Token::LedgerEntryStart),
            '$' => Some(Token::DollarSymbol),
            '(' => Some(Token::CommentStart),
            ')' => Some(Token::CommentEnd),
            '\n' => Some(Token::LineEnd),
            ',' => Some(Token::Comma),
            _ => None,
        }
    }

    fn for_word(str: &str) -> Token {
        match str {
            "paid" => Token::KeywordPaid,
            "for" => Token::KeywordFor,
            _ => Token::Word(str.to_string()),
        }
    }
}