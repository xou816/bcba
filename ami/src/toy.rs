use std::fmt::{write, Display};

use crate::token::TokenDeserialize;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    BraceOpen,
    BraceClose,
    Comma,
    If,
    True,
    Word(String),
    LineEnd
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::BraceOpen => write!(f, "opening brace `{{`"),
            Token::BraceClose => write!(f, "closing brace `}}`"),
            Token::Comma => write!(f, "comma"),
            Token::If => write!(f, "keyword `if`"),
            Token::True => write!(f, "keyword `true`"),
            Token::Word(w) => write!(f, "token `{}`", w),
            Token::LineEnd => write!(f, "line end"),
        }
    }
}

impl Eq for Token {}

impl TokenDeserialize for Token {
    fn try_from_char(c: char) -> Option<Self> {
        match c {
            '{' => Some(Self::BraceOpen),
            '}' => Some(Self::BraceClose),
            ',' => Some(Self::Comma),
            '\n' => Some(Self::LineEnd),
            _ => None
        }
    }

    fn for_word(str: &str) -> Self {
        match str {
            "if" => Token::If,
            "true" => Token::True,
            w => Token::Word(w.to_string())
        }
    }
}