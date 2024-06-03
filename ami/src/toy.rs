use std::fmt::Display;

use crate::token::{TokenDeserialize, TokenProducer};

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
    Identifier(String),
    LineEnd,
    LitString(String),
    LitNum(f32)
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
        }
    }
}

impl Eq for Token {}

impl TokenProducer for Token {
    type Token = Self;

    fn tokenize(word: &str, buffer: &mut crate::token::Buffer) -> Option<Self> {
        match word {
            "\"" => buffer.until_done('"', |s| Token::LitString(s)),
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
            _ => Some(Token::Identifier(word.to_string())),
        }
    }
}

impl TokenDeserialize for Token {
    fn try_from_char(c: char) -> Option<Self> {
        match c {
            '{' => Some(Self::BraceOpen),
            '}' => Some(Self::BraceClose),
            '(' => Some(Self::ParenOpen),
            ')' => Some(Self::ParenClose),
            ',' => Some(Self::Comma),
            '\n' => Some(Self::LineEnd),
            _ => None,
        }
    }

    fn for_word(str: &str) -> Self {
        match str {
            "if" => Token::If,
            "true" => Token::True,
            "false" => Token::False,
            "else" => Token::Else,
            w => Token::Identifier(w.to_string()),
        }
    }
}