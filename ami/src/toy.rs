use std::fmt::Display;

use crate::token::{Annotated, TokenProducer};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    BraceOpen,
    BraceClose,
    ParenOpen,
    ParenClose,
    Comma,
    If,
    Else,
    True,
    False,
    And,
    Assign,
    Let,
    Identifier(String),
    LineEnd,
    LitString(String),
    LitNum(f32)
}

impl Token {
    pub fn at(self, row: usize, col: usize) -> Annotated<Self> {
        Annotated {
            token: self,
            row,
            col,
        }
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::BraceOpen => write!(f, "opening brace `{{`"),
            Token::BraceClose => write!(f, "closing brace `}}`"),
            Token::ParenOpen => write!(f, "opening parenthesis `(`"),
            Token::ParenClose => write!(f, "closing parenthesis `)`"),
            Token::Comma => write!(f, "comma"),
            Token::If => write!(f, "keyword `if`"),
            Token::True => write!(f, "keyword `true`"),
            Token::Else => write!(f, "keyword `else`"),
            Token::False => write!(f, "keyword `false`"),
            Token::Identifier(w) => write!(f, "token `{}`", w),
            Token::LineEnd => write!(f, "line end"),
            Token::LitString(_) => write!(f, "string litteral"),
            Token::LitNum(_) => write!(f, "number litteral"),
            Token::And => write!(f, "operator `and`"),
            Token::Assign => write!(f, "`=`"),
            Token::Let => write!(f, "keyword `let`"),
        }
    }
}

impl Eq for Token {}

impl TokenProducer for Token {
    type Token = Self;

    fn tokenize(word: &str, buffer: &mut crate::token::Buffer) -> Option<Self> {
        match word {
            "\"" => buffer.until_done('"', |s| Token::LitString(s)),
            "&" => buffer.until_done('&', |_| Token::And),
            "{" => Some(Self::BraceOpen),
            "}" => Some(Self::BraceClose),
            "(" => Some(Self::ParenOpen),
            ")" => Some(Self::ParenClose),
            "," => Some(Self::Comma),
            "\n" => Some(Self::LineEnd),
            "if" => Some(Token::If),
            "true" => Some(Token::True),
            "false" => Some(Token::False),
            "else" => Some(Token::Else),
            "=" => Some(Token::Assign),
            "let" => Some(Token::Let),
            _ => Some(Token::Identifier(word.to_string())),
        }
    }
}