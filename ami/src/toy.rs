use std::fmt::Display;

use crate::{token::{Annotated, TokenizerV3}, tokenizers::*};

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

pub fn toy_tokenizer() -> TokenizerV3<Token> {
    TokenizerV3::new(vec![
        keyword("if", Token::If),
        keyword("true", Token::True),
        keyword("false", Token::False),
        keyword("else", Token::Else),
        keyword("=", Token::Assign),
        keyword("let", Token::Let),
        keyword("&&", Token::And),
        keyword("{", Token::BraceOpen),
        keyword("}", Token::BraceClose),
        keyword("(", Token::ParenOpen),
        keyword(")", Token::ParenClose),
        keyword(",", Token::Comma),
        keyword("\n", Token::LineEnd),
        keyword("{", Token::BraceOpen),
        identifier(Token::Identifier),
        delimited("\"", "\"", Token::LitString),
        ignore_whitespace()
    ])
}