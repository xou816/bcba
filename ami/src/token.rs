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

#[derive(Default)]
pub struct Buffer {
    buffer: String,
    pos: (usize, usize),
    until: Option<char>,
}

impl Buffer {
    pub fn done<T>(&mut self, f: impl FnOnce(String) -> T) -> Option<T> {
        self.until = None;
        Some(f(std::mem::take(&mut self.buffer)))
    }

    pub fn push(&mut self, s: &str) -> &mut Self {
        self.buffer.push_str(s);
        self
    }

    pub fn until<T>(&mut self, c: char) -> Option<T> {
        self.until = Some(c);
        None
    }

    pub fn until_done<T>(&mut self, c: char, f: impl FnOnce(String) -> T) -> Option<T> {
        match self.until {
            Some(_) => {
                self.until = None;
                Some(f(std::mem::take(&mut self.buffer)))
            }
            None => {
                self.until = Some(c);
                None
            }
        }
    }

    pub fn buffering(&self) -> bool {
        self.until.is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

pub enum Tokenize<T> {
    Buffer,
    Yield(T),
}

pub trait TokenProducer {
    type Token;
    fn tokenize(word: &str, buffer: &mut Buffer) -> Option<Self::Token>;
}

pub struct Tokenizer<P> {
    producer: PhantomData<P>,
    buffer: Buffer,
    col: usize,
    row: usize,
}

impl<P> Tokenizer<P>
where
    P: TokenProducer + 'static,
{
    pub fn new() -> Self {
        Self {
            producer: PhantomData,
            buffer: Default::default(),
            col: 1,
            row: 1,
        }
    }

    fn compute_pos(&mut self, word: &str) {
        if !self.buffer.buffering() {
            self.buffer.pos = (self.row, self.col);
        }
        match word {
            "\n" => {
                self.col = 1;
                self.row += 1;
            }
            _ => self.col += word.graphemes(true).count(),
        }
    }

    pub fn tokenize(mut self, program: &str) -> impl Iterator<Item = Annotated<P::Token>> + '_ {
        program
            .split_word_bounds()
            .into_iter()
            .filter_map(move |word| {
                self.compute_pos(word);

                let is_whitespace = word != "\n" && word.chars().all(char::is_whitespace);
                if is_whitespace && !self.buffer.buffering() {
                    return None;
                }

                if !self.buffer.buffering() || self.buffer.until == word.chars().next() {
                    let token = P::tokenize(word, &mut self.buffer)?;
                    let (row, col) = self.buffer.pos;
                    Some(Annotated { token, row, col })
                } else {
                    self.buffer.push(word);
                    None
                }
            })
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
    fn new(
        left: &'static str,
        right: &'static str,
        target: F,
    ) -> StrTokenizer<T> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toy::{toy_tokenizer, Token};


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
}
