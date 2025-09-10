use std::{fmt::Display, iter::once, marker::PhantomData};

use itertools::Itertools;
use unicode_segmentation::UnicodeSegmentation;

use crate::{
    combinators::OneOfParser,
    core::{BoxedParser, ParseResult, Parser, PeekResult},
};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Annotated<Token> {
    pub token: Token,
    pub row: usize,
    pub col: usize,
}

impl<T: Display> Annotated<T> {
    pub fn describe(&self) -> String {
        format!("{} at ln {}, col {}", self.token, self.row, self.col)
    }

    pub fn map(&mut self, f: impl FnOnce(&mut T)) -> &mut Self {
        let token = &mut self.token;
        f(token);
        self
    }
}

pub mod tokenizers {
    use super::*;

    pub fn keyword<'a, T: Clone + 'static>(
        keyword: &'static str,
        target: T,
    ) -> StrTokenizer<'a, T> {
        KeywordTokenizer::new(keyword, target)
    }

    pub fn ignore_whitespace<'a, T: Clone + 'static>() -> StrTokenizer<'a, T> {
        WhitespaceTokenizer::new()
    }

    pub fn delimited<'a, F, T>(
        left: &'static str,
        right: &'static str,
        target: F,
    ) -> StrTokenizer<'a, T>
    where
        F: Fn(String) -> T,
        T: 'static,
        F: 'static,
    {
        DelimitedTokenizer::new(left, right, target)
    }

    pub fn identifier<'a, F, T>(target: F) -> StrTokenizer<'a, T>
    where
        F: Fn(String) -> T + 'static,
        T: 'static,
    {
        IdentifierTokenizer::new(target)
    }

    pub fn numeric<'a, F, T>(target: F) -> StrTokenizer<'a, T>
    where
        F: Fn(Numeric64) -> T + 'static,
        T: 'static,
    {
        SimplisticNumericTokenizer::new(target)
    }
}

/// Tokenizes a borrowed string to a single token T or None
type StrTokenizer<'a, Token> = BoxedParser<'a, &'a str, Option<Annotated<Token>>>;

pub trait Tokenizable
where
    Self: Sized,
{
    fn tokenizer<'a>() -> Tokenizer<'a, Self>;
}

pub struct Tokenizer<'a, Token> {
    rules: OneOfParser<'a, &'a str, Option<Annotated<Token>>>,
    col: usize,
    row: usize,
}

impl<'a, Token: 'static> Tokenizer<'a, Token> {
    pub fn new(rules: Vec<StrTokenizer<'a, Token>>) -> Self {
        Self {
            rules: OneOfParser::new(rules),
            col: 1,
            row: 1,
        }
    }

    fn make(&self, token: &'a str) -> Annotated<&'a str> {
        Annotated {
            token,
            row: self.row,
            col: self.col,
        }
    }

    pub fn tokenize(mut self, program: &'a str) -> impl Iterator<Item = Annotated<Token>> + 'a {
        program
            .graphemes(true)
            .map_into()
            .chain(once(None))
            .tuple_windows()
            .filter_map(move |(cur, next)| {
                let cur = cur.expect("tuple_windows guarantees it to be not None");
                let cur_is_nl = cur == "\n";
                let cur_token = self.make(&cur);
                let next_token = next.map(|n| self.make(&n));
                if cur_is_nl {
                    self.col = 1;
                    self.row += 1;
                } else {
                    self.col += 1;
                }
                match self.rules.parse(cur_token, next_token.as_ref()) {
                    ParseResult::Complete(token, _) => token,
                    ParseResult::Accepted(_) => None,
                    _ => None,
                }
            })
    }
}

struct WhitespaceTokenizer<'a, T: Clone>(PhantomData<&'a T>);

impl<'a, T: Clone + 'static> WhitespaceTokenizer<'a, T> {
    fn new() -> StrTokenizer<'a, T> {
        Self(PhantomData).boxed()
    }

    fn matches(&self, token: &Annotated<&'a str>) -> bool {
        token.token != "\n" && token.token.chars().all(char::is_whitespace)
    }
}

impl<'a, T: Clone + 'static> Parser for WhitespaceTokenizer<'a, T> {
    type Expression = Option<Annotated<T>>;
    type Token = &'a str;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        if self.matches(token) {
            PeekResult::WouldComplete
        } else {
            PeekResult::WouldFail("Not whitespace".to_string())
        }
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        _: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        if self.matches(&token) {
            ParseResult::Complete(None, None)
        } else {
            ParseResult::Failed("Not whitespace".to_string(), token)
        }
    }

    fn reset(&mut self) {}
}

struct KeywordTokenizer<'a, T> {
    i: usize,
    keyword: Vec<String>,
    target: T,
    pos: Option<(usize, usize)>,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T> KeywordTokenizer<'a, T>
where
    T: Clone + 'static,
{
    fn new(keyword: &'static str, target: T) -> StrTokenizer<'a, T> {
        Self {
            i: 0,
            keyword: keyword.graphemes(true).map(|s| s.to_owned()).collect(),
            target,
            pos: None,
            _lifetime: PhantomData,
        }
        .boxed()
    }

    fn ith_token_matches(&self, i: usize, token: &Annotated<&'a str>) -> bool {
        if i >= self.keyword.len() {
            return false;
        }

        self.keyword
            .get(i)
            .map(|i| i == &token.token)
            .unwrap_or_default()
    }

    fn is_last(&self) -> bool {
        self.i + 1 == self.keyword.len()
    }

    fn make(&self) -> Annotated<T> {
        let (row, col) = self.pos.unwrap();
        Annotated {
            token: self.target.clone(),
            row,
            col,
        }
    }
}

impl<'a, T> Parser for KeywordTokenizer<'a, T>
where
    T: Clone + 'static,
{
    type Expression = Option<Annotated<T>>;
    type Token = &'a str;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        let matches = self.ith_token_matches(self.i, token);
        if matches && self.is_last() {
            PeekResult::WouldComplete
        } else if matches {
            PeekResult::WouldAccept
        } else {
            PeekResult::WouldFail("Not expected keyword".to_string())
        }
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        next: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        if self.pos.is_none() {
            self.pos = Some((token.row, token.col));
        }
        let matches = self.ith_token_matches(self.i, &token);
        let next_matches = next
            .map(|n| self.ith_token_matches(self.i + 1, n))
            .unwrap_or(false);
        let is_last = self.is_last();
        match (matches, next_matches, is_last) {
            (true, _, true) => self.complete(Some(self.make()), None),
            (true, true, _) => {
                self.i += 1;
                ParseResult::Accepted(None)
            }
            _ => self.fail_token("Not expected keyword", token),
        }
    }

    fn reset(&mut self) {
        self.i = 0;
        self.pos = None;
    }
}

struct DelimitedTokenizer<'a, T, F>
where
    F: Fn(String) -> T,
{
    parsing_inner: bool,
    left: &'static str,
    right: &'static str,
    target: F,
    acc: String,
    pos: Option<(usize, usize)>,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T, F> DelimitedTokenizer<'a, T, F>
where
    F: Fn(String) -> T,
    T: 'static,
    F: 'static,
{
    fn new(left: &'static str, right: &'static str, target: F) -> StrTokenizer<'a, T> {
        Self {
            parsing_inner: false,
            left,
            right,
            target,
            acc: String::new(),
            pos: None,
            _lifetime: PhantomData,
        }
        .boxed()
    }

    fn make(&mut self) -> Annotated<T> {
        let (row, col) = self.pos.take().unwrap();
        Annotated {
            token: (self.target)(std::mem::take(&mut self.acc)),
            row,
            col,
        }
    }
}

impl<'a, T, F> Parser for DelimitedTokenizer<'a, T, F>
where
    F: Fn(String) -> T,
    T: 'static,
    F: 'static,
{
    type Expression = Option<Annotated<T>>;
    type Token = &'a str;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        if self.parsing_inner && token.token == self.right {
            PeekResult::WouldComplete
        } else if self.parsing_inner || token.token == self.left {
            PeekResult::WouldAccept
        } else {
            PeekResult::WouldFail("Expected left delimiter".to_string())
        }
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        _: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        if self.pos.is_none() {
            self.pos = Some((token.row, token.col));
        }

        if !self.parsing_inner {
            let token_ok = token.token == self.left;
            if token_ok {
                self.parsing_inner = true;
                ParseResult::Accepted(None)
            } else {
                self.fail_token("Expected left delimiter", token)
            }
        } else {
            let is_finishing = token.token == self.right;
            if !is_finishing {
                self.acc.push_str(&token.token);
                ParseResult::Accepted(None)
            } else {
                let t = self.make();
                self.complete(Some(t), None)
            }
        }
    }

    fn reset(&mut self) {
        self.acc = String::new();
        self.parsing_inner = false;
        self.pos = None;
    }
}

struct IdentifierTokenizer<'a, T, F2>
where
    F2: Fn(String) -> T,
{
    expect: Box<dyn Fn(&'a str) -> bool>,
    target: F2,
    acc: String,
    pos: Option<(usize, usize)>,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T, F2> IdentifierTokenizer<'a, T, F2>
where
    F2: Fn(String) -> T + 'static,
    T: 'static,
{
    fn new(target: F2) -> StrTokenizer<'a, T> {
        Self {
            expect: Box::new(|s| s.chars().all(|c| c.is_alphabetic() || c == '_')),
            target,
            acc: String::new(),
            pos: None,
            _lifetime: PhantomData,
        }
        .boxed()
    }

    fn make(&mut self) -> Annotated<T> {
        let (row, col) = self.pos.take().unwrap();
        Annotated {
            token: (self.target)(std::mem::take(&mut self.acc)),
            row,
            col,
        }
    }
}

impl<'a, T, F2> Parser for IdentifierTokenizer<'a, T, F2>
where
    F2: Fn(String) -> T + 'static,
    T: 'static,
{
    type Expression = Option<Annotated<T>>;
    type Token = &'a str;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        let token_matches = (self.expect)(&token.token);
        let started = !self.acc.is_empty();
        match (token_matches, started) {
            (true, _) => PeekResult::WouldAccept,
            (false, false) => PeekResult::WouldFail("Unexpected identifier".to_string()),
            (false, true) => PeekResult::WouldComplete,
        }
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        next: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        if self.pos.is_none() {
            self.pos = Some((token.row, token.col));
        }

        let token_matches = (self.expect)(&token.token);
        let next_token_matches = next.map(|n| (self.expect)(&n.token)).unwrap_or(false);
        match (token_matches, next_token_matches) {
            (true, true) => {
                self.acc.push_str(&token.token);
                ParseResult::Accepted(None)
            }
            (true, false) => {
                self.acc.push_str(&token.token);
                let t = self.make();
                self.complete(Some(t), None)
            }
            (false, _) => self.fail_token("Unexpected identifier", token),
        }
    }

    fn reset(&mut self) {
        self.pos = None;
        self.acc = Default::default();
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Numeric64 {
    Int(i64),
    Float(f64),
}

impl PartialEq for Numeric64 {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Float(l0), Self::Float(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl Eq for Numeric64 {}

enum NumericElement {
    Dot,
    MinusSign,
    Digit(u32),
}

impl TryFrom<&'_ str> for NumericElement {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let char = value
            .chars()
            .exactly_one()
            .map_err(|_| "Unexpected char".to_string())?;
        if char.is_ascii_digit() {
            return Ok(Self::Digit(char as u32 - '0' as u32));
        }
        match char {
            '.' => Ok(Self::Dot),
            '-' => Ok(Self::MinusSign),
            _ => Err("Unexpected char".to_string()),
        }
    }
}

struct SimplisticNumericTokenizer<'a, F, T>
where
    F: Fn(Numeric64) -> T,
{
    target: F,
    neg: bool,
    pre: Option<u32>,
    post: Option<f64>,
    post_exponent: f64,
    pos: Option<(usize, usize)>,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, F, T> SimplisticNumericTokenizer<'a, F, T>
where
    F: Fn(Numeric64) -> T + 'static,
    T: 'static,
{
    fn new(target: F) -> StrTokenizer<'a, T> {
        Self {
            target,
            neg: false,
            pre: None,
            post: None,
            post_exponent: 1.0,
            pos: None,
            _lifetime: PhantomData,
        }
        .boxed()
    }

    fn make(&mut self) -> Annotated<T> {
        let (row, col) = self.pos.take().unwrap();
        match (self.pre, self.post) {
            (Some(pre), None) => {
                let pre: i64 = pre.into();
                let neg = if self.neg { -1i64 } else { 1i64 };
                let numeric = Numeric64::Int(neg * pre);
                Annotated {
                    token: (self.target)(numeric),
                    row,
                    col,
                }
            }
            (Some(pre), Some(post)) => {
                let pre: f64 = pre.into();
                let neg = if self.neg { -1f64 } else { 1f64 };
                let numeric = Numeric64::Float(neg * (pre + post));
                Annotated {
                    token: (self.target)(numeric),
                    row,
                    col,
                }
            }
            _ => unreachable!("wtf?"),
        }
    }
}

impl<'a, F, T> Parser for SimplisticNumericTokenizer<'a, F, T>
where
    F: Fn(Numeric64) -> T + 'static,
    T: 'static,
{
    type Expression = Option<Annotated<T>>;
    type Token = &'a str;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        match NumericElement::try_from(token.token) {
            Ok(_) => PeekResult::WouldAccept,
            Err(e) => PeekResult::WouldFail(e),
        }
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        if self.pos.is_none() {
            self.pos = Some((token.row, token.col));
        }

        let el = match NumericElement::try_from(token.token) {
            Ok(el) => el,
            Err(e) => return self.fail_token(&e, token),
        };
        let next_token = next_token.and_then(|t| NumericElement::try_from(t.token).ok());
        match (el, next_token, self.pre, self.post) {
            (NumericElement::MinusSign, Some(NumericElement::Digit(_)), None, None) => {
                self.neg = true;
                self.pre = Some(0);
                ParseResult::Accepted(None)
            }
            (
                NumericElement::Digit(d),
                Some(NumericElement::Digit(_) | NumericElement::Dot),
                _,
                None,
            ) => {
                self.pre = Some(self.pre.unwrap_or_default() * 10 + d);
                ParseResult::Accepted(None)
            }
            (NumericElement::Digit(d), None, _, None) => {
                self.pre = Some(self.pre.unwrap_or_default() * 10 + d);
                ParseResult::Complete(Some(self.make()), None)
            }
            (NumericElement::Dot, _, Some(_), None) => {
                self.post = Some(0.0);
                ParseResult::Accepted(None)
            }
            (NumericElement::Digit(d), Some(NumericElement::Digit(_)), Some(_), Some(post)) => {
                self.post_exponent *= 0.1;
                self.post = Some(post + (d as f64) * self.post_exponent);
                ParseResult::Accepted(None)
            }
            (NumericElement::Digit(d), None, Some(_), Some(post)) => {
                self.post_exponent *= 0.1;
                self.post = Some(post + (d as f64) * self.post_exponent);
                ParseResult::Complete(Some(self.make()), None)
            }

            _ => self.fail_token("Unexpected numeric value", token),
        }
    }

    fn reset(&mut self) {
        self.pos = None;
        self.neg = false;
        self.pre = None;
        self.post = None;
        self.post_exponent = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toy::{toy_tokenizer, Token};

    fn make_line(t: &str) -> impl Iterator<Item = Annotated<&'_ str>> + '_ {
        t.grapheme_indices(true).map(|(i, s)| Annotated {
            token: s,
            row: 1,
            col: i + 1,
        })
    }

    #[test]
    fn test_tokenizer() {
        let tokens: Vec<Annotated<Token>> = toy_tokenizer()
            .tokenize("if true {\nprint(\"hellô  world\")\n}")
            .collect();
        let expected = vec![
            Token::If.at(1, 1),
            Token::True.at(1, 4),
            Token::BraceOpen.at(1, 9),
            Token::LineEnd.at(1, 10),
            Token::Identifier("print".to_owned()).at(2, 1),
            Token::ParenOpen.at(2, 6),
            Token::LitString("hellô  world".to_owned()).at(2, 7),
            Token::ParenClose.at(2, 21),
            Token::LineEnd.at(2, 22),
            Token::BraceClose.at(3, 1),
        ];
        assert_eq!(expected, tokens);
    }

    #[test]
    fn test_numeric() {
        let mut tokenizer = SimplisticNumericTokenizer::new(Token::LitNum);

        let mut tokens = make_line("-12.989");
        let res = tokenizer.run_to_completion(&mut tokens);
        assert_eq!(
            res,
            Ok(Some(Token::LitNum(Numeric64::Float(-12.9890)).at(1, 1)))
        )
    }

    #[test]
    fn test_numeric2() {
        let mut tokenizer = SimplisticNumericTokenizer::new(Token::LitNum);

        let mut tokens = make_line("1.01");
        let res = tokenizer.run_to_completion(&mut tokens);
        assert_eq!(
            res,
            Ok(Some(Token::LitNum(Numeric64::Float(1.01)).at(1, 1)))
        )
    }

    #[test]
    fn test_numeric3() {
        let mut tokenizer = SimplisticNumericTokenizer::new(Token::LitNum);

        let mut tokens = make_line("123");
        let res = tokenizer.run_to_completion(&mut tokens);
        assert_eq!(res, Ok(Some(Token::LitNum(Numeric64::Int(123)).at(1, 1))))
    }
}
