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
}

pub mod tokenizers {
    use super::*;

    pub fn keyword<T: Clone + 'static>(keyword: &str, target: T) -> StrTokenizer<T> {
        KeywordTokenizer::new(keyword, target)
    }

    pub fn ignore_whitespace<T: 'static>() -> StrTokenizer<T> {
        WhitespaceTokenizer::new()
    }

    pub fn delimited<F, T>(left: &'static str, right: &'static str, target: F) -> StrTokenizer<T>
    where
        F: Fn(String) -> T,
        T: 'static,
        F: 'static,
    {
        DelimitedTokenizer::new(left, right, target)
    }

    pub fn identifier<F, T>(target: F) -> StrTokenizer<T>
    where
        F: Fn(String) -> T + 'static,
        T: 'static,
    {
        IdentifierTokenizer::new(target)
    }

    pub fn numeric<F, T>(target: F) -> StrTokenizer<T>
    where
        F: Fn(Numeric64) -> T + 'static,
        T: 'static,
    {
        SimplisticNumericTokenizer::new(target)
    }
}

type StrTokenizer<T> = BoxedParser<String, Option<Annotated<T>>>;

pub struct TokenizerV3<Token> {
    rules: OneOfParser<String, Option<Annotated<Token>>>,
    col: usize,
    row: usize,
}

impl<Token: 'static> TokenizerV3<Token> {
    pub fn new(rules: Vec<StrTokenizer<Token>>) -> Self {
        Self {
            rules: OneOfParser::new(rules),
            col: 1,
            row: 1,
        }
    }

    fn make(&self, token: String) -> Annotated<String> {
        Annotated {
            token,
            row: self.row,
            col: self.col,
        }
    }

    pub fn tokenize(mut self, program: &'_ str) -> impl Iterator<Item = Annotated<Token>> + '_ {
        program
            .graphemes(true)
            .map(|g| Some(g.to_owned()))
            .chain(once(None))
            .tuple_windows()
            .filter_map(move |(cur, next)| {
                let cur = cur.expect("tuple_windows guarantees it to be not None");
                let cur_is_nl = cur == "\n";
                let cur_token = self.make(cur);
                let next_token = next.map(|n| self.make(n));
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

struct WhitespaceTokenizer<T>(PhantomData<T>);

impl<T: 'static> WhitespaceTokenizer<T> {
    fn new() -> StrTokenizer<T> {
        Self(PhantomData).boxed()
    }

    fn matches(&self, token: &Annotated<String>) -> bool {
        token.token != "\n" && token.token.chars().all(char::is_whitespace)
    }
}

impl<T: 'static> Parser for WhitespaceTokenizer<T> {
    type Expression = Option<Annotated<T>>;
    type Token = String;

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

struct KeywordTokenizer<T> {
    i: usize,
    keyword: Vec<String>,
    target: T,
    pos: Option<(usize, usize)>,
}

impl<T> KeywordTokenizer<T>
where
    T: Clone + 'static,
{
    fn new(keyword: &str, target: T) -> StrTokenizer<T> {
        Self {
            i: 0,
            keyword: keyword.graphemes(true).map(|s| s.to_owned()).collect(),
            target,
            pos: None,
        }
        .boxed()
    }

    fn ith_token_matches(&self, i: usize, token: &Annotated<String>) -> bool {
        if i >= self.keyword.len() {
            return false;
        }

        Some(&token.token) == self.keyword.get(i)
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

impl<T> Parser for KeywordTokenizer<T>
where
    T: Clone + 'static,
{
    type Expression = Option<Annotated<T>>;
    type Token = String;

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
            (true, _, true) => self.complete(Some(self.make())),
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

struct DelimitedTokenizer<T, F>
where
    F: Fn(String) -> T,
{
    parsing_inner: bool,
    left: &'static str,
    right: &'static str,
    target: F,
    acc: String,
    pos: Option<(usize, usize)>,
}

impl<T, F> DelimitedTokenizer<T, F>
where
    F: Fn(String) -> T,
    T: 'static,
    F: 'static,
{
    fn new(left: &'static str, right: &'static str, target: F) -> StrTokenizer<T> {
        Self {
            parsing_inner: false,
            left,
            right,
            target,
            acc: String::new(),
            pos: None,
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

impl<T, F> Parser for DelimitedTokenizer<T, F>
where
    F: Fn(String) -> T,
    T: 'static,
    F: 'static,
{
    type Expression = Option<Annotated<T>>;
    type Token = String;

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
                self.complete(Some(t))
            }
        }
    }

    fn reset(&mut self) {
        self.acc = String::new();
        self.parsing_inner = false;
        self.pos = None;
    }
}

struct IdentifierTokenizer<T, F2>
where
    F2: Fn(String) -> T,
{
    expect: Box<dyn Fn(&String) -> bool>,
    target: F2,
    acc: String,
    pos: Option<(usize, usize)>,
}

impl<T, F2> IdentifierTokenizer<T, F2>
where
    F2: Fn(String) -> T + 'static,
    T: 'static,
{
    fn new(target: F2) -> StrTokenizer<T> {
        Self {
            expect: Box::new(|s| s.chars().all(|c| c.is_alphabetic() || c == '_')),
            target,
            acc: String::new(),
            pos: None,
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

impl<T, F2> Parser for IdentifierTokenizer<T, F2>
where
    F2: Fn(String) -> T + 'static,
    T: 'static,
{
    type Expression = Option<Annotated<T>>;
    type Token = String;

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
                self.complete(Some(t))
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

struct SimplisticNumericTokenizer<F, T>
where
    F: Fn(Numeric64) -> T,
{
    target: F,
    neg: bool,
    pre: Option<u32>,
    post: Option<u32>,
    pos: Option<(usize, usize)>,
}

impl<F, T> SimplisticNumericTokenizer<F, T>
where
    F: Fn(Numeric64) -> T + 'static,
    T: 'static,
{
    fn new(target: F) -> StrTokenizer<T> {
        Self {
            target,
            neg: false,
            pre: None,
            post: None,
            pos: None,
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
                let post: f64 = post.into();
                let exp = (post.log10()).floor() as i32;
                let neg = if self.neg { -1f64 } else { 1f64 };
                let numeric = Numeric64::Float(neg * (pre + post * 10f64.powi(-exp - 1)));
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

impl<F, T> Parser for SimplisticNumericTokenizer<F, T>
where
    F: Fn(Numeric64) -> T + 'static,
    T: 'static,
{
    type Expression = Option<Annotated<T>>;
    type Token = String;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        match NumericElement::try_from(token.token.as_str()) {
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

        let el = match NumericElement::try_from(token.token.as_str()) {
            Ok(el) => el,
            Err(e) => return self.fail_token(&e, token),
        };
        let next_token = next_token.and_then(|t| NumericElement::try_from(t.token.as_str()).ok());
        match (el, next_token, self.pre, self.post) {
            (NumericElement::Dot, _, Some(_), None) => {
                self.post = Some(0);
                ParseResult::Accepted(None)
            }
            (NumericElement::MinusSign, Some(NumericElement::Digit(_)), None, None) => {
                self.neg = true;
                self.pre = Some(0);
                ParseResult::Accepted(None)
            }
            (NumericElement::Digit(d), Some(NumericElement::Digit(_)), Some(_), Some(post)) => {
                self.post = Some(post * 10 + d);
                ParseResult::Accepted(None)
            }
            (NumericElement::Digit(d), None, Some(_), Some(post)) => {
                self.post = Some(post * 10 + d);
                ParseResult::Complete(Some(self.make()), None)
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
            (
                NumericElement::Digit(_),
                None,
                _,
                None,
            ) => ParseResult::Complete(Some(self.make()), None),

            _ => self.fail_token("Unexpected numeric value", token),
        }
    }

    fn reset(&mut self) {
        self.pos = None;
        self.neg = false;
        self.pre = None;
        self.post = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toy::{toy_tokenizer, Token};

    fn make_line(t: &str) -> impl Iterator<Item = Annotated<String>> + '_ {
        t.grapheme_indices(true).map(|(i, s)| Annotated {
            token: s.to_owned(),
            row: 1,
            col: i + 1,
        })
    }

    #[test]
    fn test_v3() {
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
        assert_eq!(res, Ok(Some(Token::LitNum(Numeric64::Float(-12.989)).at(1, 1))))
    }
}
