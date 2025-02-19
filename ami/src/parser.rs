#![allow(dead_code)]
use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    vec,
};

use super::token::Annotated;

#[derive(Debug)]
pub enum ParseResult<Token, Result> {
    Accepted(Option<Annotated<Token>>),
    Complete(Result, Option<Annotated<Token>>),
    Failed(String, Option<Annotated<Token>>),
}

impl<T, Result> ParseResult<T, Result> {
    fn map_result<F, R2>(self, f: F) -> ParseResult<T, R2>
    where
        F: FnOnce(Result) -> R2,
    {
        match self {
            ParseResult::Complete(r, t) => ParseResult::Complete(f(r), t),
            ParseResult::Accepted(t) => ParseResult::Accepted(t),
            ParseResult::Failed(e, t) => ParseResult::Failed(e, t),
        }
    }
}

pub trait Peek {
    type Token: Display;

    fn peek(&self, token: &Annotated<Self::Token>) -> bool;
}

pub trait Parser {
    type Expression;
    type Token: Display;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression>;

    fn run_to_completion(
        &mut self,
        tokens: &mut dyn Iterator<Item = Annotated<Self::Token>>,
    ) -> Result<Self::Expression, String> {
        let mut tokens = tokens.peekable();
        let mut next_token: Option<Annotated<Self::Token>> = None;
        loop {
            next_token = next_token.or(tokens.next());
            let Some(token) = next_token.take() else {
                break Err("Unexpected end of input: too few tokens to complete".to_string());
            };
            let peeked_token = tokens.peek();
            match self.try_consume(token, peeked_token) {
                ParseResult::Complete(res, _) => break Ok(res),
                ParseResult::Failed(err, _) => break Err(err),
                ParseResult::Accepted(t) => {
                    next_token = t;
                    continue;
                }
            }
        }
    }

    fn run_to_exhaustion(
        &mut self,
        tokens: &mut dyn Iterator<Item = Annotated<Self::Token>>,
    ) -> Result<Vec<Self::Expression>, String> {
        let mut tokens = tokens.peekable();
        let mut next_token: Option<Annotated<Self::Token>> = None;
        let mut acc: Vec<Self::Expression> = vec![];
        loop {
            next_token = next_token.or(tokens.next());
            let Some(token) = next_token.take() else {
                break Ok(acc);
            };
            let next = tokens.peek();
            match self.try_consume(token, next) {
                ParseResult::Complete(res, t) => {
                    next_token = t;
                    acc.push(res)
                }
                ParseResult::Failed(err, _) => break Err(err),
                ParseResult::Accepted(_) if next.is_none() => {
                    break Err("Unexpected end of input: parsing interupted".to_string())
                }
                ParseResult::Accepted(t) => {
                    next_token = t;
                    continue;
                }
            }
        }
    }

    fn as_peek(&self) -> Option<Box<dyn Peek<Token = Self::Token> + '_>> {
        None
    }

    fn reset(&mut self);

    fn complete(&mut self, r: Self::Expression) -> ParseResult<Self::Token, Self::Expression> {
        self.reset();
        ParseResult::Complete(r, None)
    }

    fn fail(
        &mut self,
        message: String,
        token: Option<Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        self.reset();
        ParseResult::Failed(message, token)
    }

    fn fail_token(
        &mut self,
        message: &str,
        token: Annotated<Self::Token>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        self.reset();
        ParseResult::Failed(
            format!("{}: unexpected {}", message, token.describe()),
            Some(token),
        )
    }

    fn map<F, U>(self, f: F) -> MapParser<Self, U>
    where
        F: Fn(Self::Expression) -> U + 'static,
        Self: Sized,
    {
        MapParser(self, Box::new(move |e| Ok(f(e))))
    }

    fn try_map<F, U>(self, f: F) -> MapParser<Self, U>
    where
        F: Fn(Self::Expression) -> Result<U, String> + 'static,
        Self: Sized,
    {
        MapParser(self, Box::new(f))
    }

    fn then<P: Parser>(self, p: P) -> ThenParser<Self, P>
    where
        Self: Sized,
    {
        ThenParser {
            cur: self,
            cur_result: None,
            next: p,
        }
    }

    fn boxed(self) -> Box<dyn Parser<Expression = Self::Expression, Token = Self::Token>>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

pub struct LazyParser<P>
where
    P: Parser,
{
    parser: Option<P>,
    get: Box<dyn Fn() -> P>,
}

impl<P> Parser for LazyParser<P>
where
    P: Parser,
    P::Token: Display,
{
    type Expression = P::Expression;
    type Token = P::Token;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let mut parser = self.parser.take().unwrap_or_else(&self.get);
        let result = parser.try_consume(token, next_token);
        self.parser.replace(parser);
        result
    }

    fn reset(&mut self) {
        self.parser = None;
    }
}

pub struct ThenParser<A, B>
where
    A: Parser,
{
    cur: A,
    cur_result: Option<A::Expression>,
    next: B,
}

impl<A, B> Parser for ThenParser<A, B>
where
    A: Parser,
    B: Parser<Token = A::Token>,
{
    type Expression = (A::Expression, B::Expression);
    type Token = A::Token;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        if self.cur_result.is_some() {
            self.next
                .try_consume(token, next_token)
                .map_result(|b| (self.cur_result.take().unwrap(), b))
        } else {
            let peek_ok = || match (self.next.as_peek(), next_token) {
                (Some(peek), Some(t)) => peek.peek(t),
                (None, _) => true,
                _ => false,
            };
            match self.cur.try_consume(token, next_token) {
                ParseResult::Accepted(t) => ParseResult::Accepted(t),
                ParseResult::Complete(a, t) if peek_ok() => {
                    self.cur_result = Some(a);
                    ParseResult::Accepted(t)
                }
                ParseResult::Complete(_, t) => {
                    let desc = next_token
                        .map(|t| t.describe())
                        .unwrap_or("token".to_string());
                    self.fail(format!("Failed to parse, unexpected {desc}"), t)
                }
                ParseResult::Failed(e, t) => ParseResult::Failed(e, t),
            }
        }
    }

    fn reset(&mut self) {
        self.cur.reset();
        self.next.reset();
        self.cur_result = None;
    }
}

pub mod parsers {

    use super::*;

    pub fn lazy<P: Parser + 'static>(p: impl Fn() -> P + 'static) -> LazyParser<P> {
        LazyParser {
            parser: None,
            get: Box::new(p),
        }
    }

    pub fn one_of<T, E>(parsers: impl IntoIterator<Item = BoxedParser<T, E>>) -> OneOfParser<T, E> {
        OneOfParser(
            parsers
                .into_iter()
                .map(|parser| CandidateParser {
                    is_candidate: true,
                    parser,
                })
                .collect(),
        )
    }

    pub fn always_completing<T, E, F>(result_factory: F) -> CompletingParser<T, E>
    where
        F: Fn() -> E + 'static,
    {
        CompletingParser(PhantomData, Box::new(result_factory))
    }

    pub fn sequence<T>(
        seq: impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T>>,
    ) -> SequenceParser<T> {
        SequenceParser {
            seq: seq.into_iter().collect(),
            i: 0,
            collected: vec![],
        }
    }

    pub fn list_of<P>(separator: P::Token, item_parser: P) -> ListParser<P>
    where
        P: Parser,
    {
        ListParser::new(separator, item_parser)
    }

    pub fn repeat_until<P>(end: P::Token, parser: P) -> RepeatUntil<P>
    where
        P: Parser,
    {
        RepeatUntil {
            parser,
            busy: false,
            end,
            acc: vec![],
        }
    }

    #[deprecated]
    pub fn discard_delimited<T>(delimiter: [T; 2]) -> DelimitedParser<DiscardUntil<T>>
    where
        T: Display + Clone + Eq + 'static,
    {
        let end = delimiter[1].clone();
        DelimitedParser {
            delimiter,
            inner_result: None,
            inner_parser: DiscardUntil(end),
            started: false,
        }
    }
}

pub struct SequenceParser<T> {
    seq: Vec<T>,
    i: usize,
    collected: Vec<T>,
}

impl<T> Parser for SequenceParser<T>
where
    T: Display + Eq,
{
    type Expression = Vec<T>;
    type Token = T;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let Self { seq, i, collected } = self;
        let Some(cur) = seq.get(*i) else {
            return ParseResult::Failed(
                "Expected to parse at least one token".to_string(),
                Some(token),
            );
        };
        let next = seq.get(*i + 1);
        let success = &token.token == cur && next_token.map(|t| &t.token) == next || next == None;
        match (success, next) {
            (true, None) => {
                collected.push(token.token);
                let r = ParseResult::Complete(collected.drain(..).collect(), None);
                self.reset();
                r
            }
            (true, Some(_)) => {
                *i += 1;
                collected.push(token.token);
                ParseResult::Accepted(None)
            }
            (false, _) => {
                let desc = next_token
                    .map(|t| t.describe())
                    .unwrap_or("token".to_string());
                let expected = next.map(|n| format!(", expected {n}")).unwrap_or_default();
                self.fail(
                    format!("Failed to parse sequence: unexpected {desc}{expected}"),
                    Some(token),
                )
            }
        }
    }

    fn reset(&mut self) {
        self.i = 0;
        self.collected = vec![];
    }
}

pub struct SingleParser<T, E> {
    token_matches: Box<dyn Fn(&Annotated<T>) -> bool>,
    map_match: Box<dyn Fn(Annotated<T>) -> Result<E, Annotated<T>>>,
}

impl<T, E> SingleParser<T, E> {
    pub fn new(
        token_matches: impl Fn(&Annotated<T>) -> bool + 'static,
        map_match: impl Fn(Annotated<T>) -> Result<E, Annotated<T>> + 'static,
    ) -> Self {
        Self {
            token_matches: Box::new(token_matches),
            map_match: Box::new(map_match),
        }
    }
}

impl<T: Display, E> Peek for &SingleParser<T, E> {
    type Token = T;

    fn peek(&self, token: &Annotated<Self::Token>) -> bool {
        (self.token_matches)(&token)
    }
}

impl<T: Display, E> Parser for SingleParser<T, E> {
    type Expression = E;
    type Token = T;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        _: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let map_match = &self.map_match;
        match map_match(token) {
            Ok(r) => self.complete(r),
            Err(t) => self.fail_token("Failed to parse single token", t),
        }
    }

    fn as_peek(&self) -> Option<Box<dyn Peek<Token = Self::Token> + '_>> {
        Some(Box::new(self))
    }

    fn reset(&mut self) {}
}

pub struct CompletingParser<T, E>(PhantomData<T>, Box<dyn Fn() -> E>);

impl<T: Display, E> Parser for CompletingParser<T, E> {
    type Expression = E;
    type Token = T;

    fn try_consume(
        &mut self,
        _: Annotated<Self::Token>,
        _: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        ParseResult::Complete(self.1(), None)
    }

    fn reset(&mut self) {}
}

pub type BoxedParser<T, E> = Box<dyn Parser<Token = T, Expression = E>>;

impl<T: Display, E> Parser for BoxedParser<T, E> {
    type Expression = E;
    type Token = T;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        self.as_mut().try_consume(token, next_token)
    }

    fn reset(&mut self) {
        self.as_mut().reset()
    }
}

pub struct MapParser<P, U>(P, Box<dyn Fn(P::Expression) -> Result<U, String>>)
where
    P: Parser + Sized;

impl<P, U> Parser for MapParser<P, U>
where
    P: Parser + Sized,
{
    type Expression = U;
    type Token = P::Token;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        match self.0.try_consume(token, next_token) {
            ParseResult::Complete(r, t) => match self.1(r) {
                Ok(r) => self.complete(r),
                Err(msg) => self.fail_token(&msg, t.expect("Fix this!")),
            },
            ParseResult::Accepted(t) => ParseResult::Accepted(t),
            ParseResult::Failed(e, t) => self.fail(e, t),
        }
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

pub struct CandidateParser<T, E> {
    is_candidate: bool,
    parser: BoxedParser<T, E>,
}
pub struct OneOfParser<T, E>(Vec<CandidateParser<T, E>>);

impl<T, E> OneOfParser<T, E> {
    fn new(parsers: Vec<BoxedParser<T, E>>) -> Self {
        Self(
            parsers
                .into_iter()
                .map(|parser| CandidateParser {
                    is_candidate: true,
                    parser,
                })
                .collect(),
        )
    }
}

impl<T, E> Parser for OneOfParser<T, E>
where
    T: Display + Clone,
{
    type Expression = E;
    type Token = T;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let result = self.0.iter_mut().filter(|p| p.is_candidate).fold(
            ParseResult::Failed("Unreachable".to_string(), Some(token.clone())),
            |res, p| match res {
                ParseResult::Failed(_, failed_token) => {
                    let t = failed_token.unwrap_or_else(|| token.clone());
                    match p.parser.try_consume(t, next_token) {
                        r @ ParseResult::Accepted(_) => r,
                        r => {
                            p.is_candidate = false;
                            r
                        }
                    }
                }
                _ => {
                    p.is_candidate = false;
                    res
                }
            },
        );
        if matches!(result, ParseResult::Complete(..) | ParseResult::Failed(..)) {
            self.reset();
        }
        result
    }

    fn reset(&mut self) {
        self.0.iter_mut().for_each(|it| {
            it.is_candidate = true;
            it.parser.reset();
        });
    }
}

pub struct DiscardUntil<T>(T);

impl<T> Parser for DiscardUntil<T>
where
    T: Display + Eq,
{
    type Expression = ();
    type Token = T;

    fn try_consume(
        &mut self,
        _: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        match next_token {
            Some(Annotated { token, .. }) if token == &self.0 => ParseResult::Complete((), None),
            _ => ParseResult::Accepted(None),
        }
    }

    fn reset(&mut self) {}
}

pub struct RepeatUntil<P: Parser> {
    parser: P,
    busy: bool,
    end: P::Token,
    acc: Vec<P::Expression>,
}

impl<P> Parser for RepeatUntil<P>
where
    P: Parser,
    P::Token: Display + Eq,
{
    type Expression = Vec<P::Expression>;
    type Token = P::Token;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let next_is_end = next_token.map(|t| t.token == self.end).unwrap_or(false);
        match self.parser.try_consume(token, next_token) {
            ParseResult::Accepted(t) => {
                self.busy = true;
                ParseResult::Accepted(t)
            }
            ParseResult::Complete(e, _) if next_is_end => {
                self.acc.push(e);
                let r = ParseResult::Complete(self.acc.drain(..).collect(), None);
                self.reset();
                r
            }
            ParseResult::Complete(e, _) => {
                self.busy = false;
                self.acc.push(e);
                ParseResult::Accepted(None)
            }
            ParseResult::Failed(e, t) => ParseResult::Failed(e, t),
        }
    }

    fn reset(&mut self) {
        self.busy = false;
        self.acc = vec![];
    }
}

pub struct DelimitedParser<P>
where
    P: Parser,
{
    delimiter: [P::Token; 2],
    inner_result: Option<P::Expression>,
    inner_parser: P,
    started: bool,
}

impl<P> Parser for DelimitedParser<P>
where
    P: Parser,
    P::Token: Display + Eq + Debug,
{
    type Expression = Option<P::Expression>;
    type Token = P::Token;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let [ref start, ref end] = self.delimiter;
        let inner_done = self.inner_result.is_some();
        let inner_next = if next_token.map(|t| &t.token == end).unwrap_or(true) {
            None
        } else {
            next_token
        };
        match (
            &token.token == start,
            &token.token == end,
            self.started,
            inner_done,
        ) {
            (true, _, false, _) => {
                self.started = true;
                ParseResult::Accepted(None)
            }
            (false, false, true, false) => match self.inner_parser.try_consume(token, inner_next) {
                ParseResult::Complete(r, t) => {
                    self.inner_result = Some(r);
                    ParseResult::Accepted(t)
                }
                r => r.map_result(Option::Some),
            },
            (_, true, _, _) => {
                let result = ParseResult::Complete(self.inner_result.take(), None);
                self.reset();
                result
            }
            _ => self.fail_token("Failed to parse delimited expression", token),
        }
    }

    fn reset(&mut self) {
        self.inner_result = None;
        self.inner_parser.reset();
        self.started = false;
    }
}

pub struct ListParser<P>
where
    P: Parser,
{
    separator: P::Token,
    item_parser: P,
    acc: Vec<P::Expression>,
}

impl<P> ListParser<P>
where
    P: Parser,
{
    fn new(separator: P::Token, item_parser: P) -> Self {
        Self {
            separator,
            item_parser,
            acc: vec![],
        }
    }
}

impl<P> Parser for ListParser<P>
where
    P: Parser,
    P::Token: Display + Eq,
{
    type Expression = Vec<P::Expression>;
    type Token = P::Token;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let is_separator = token.token == self.separator;
        let next_is_end = next_token.is_none();
        let next_is_not_separator =
            matches!(next_token, Some(Annotated {token,..}) if token != &self.separator);

        match (is_separator, next_is_end) {
            (true, true) => ParseResult::Complete(self.acc.drain(..).collect(), None),
            (true, false) => ParseResult::Accepted(None),
            (false, _) => match self.item_parser.try_consume(token, next_token) {
                ParseResult::Accepted(t) => ParseResult::Accepted(t),
                ParseResult::Complete(item, t) => {
                    self.acc.push(item);
                    if next_is_end || next_is_not_separator {
                        ParseResult::Complete(self.acc.drain(..).collect(), t)
                    } else {
                        ParseResult::Accepted(t)
                    }
                }
                ParseResult::Failed(_, t) if self.acc.is_empty() => {
                    ParseResult::Complete(vec![], t)
                }
                ParseResult::Failed(_, t) => self.fail("Expected list element".to_string(), t),
            },
        }
    }

    fn reset(&mut self) {
        self.acc = vec![];
    }
}

#[macro_export]
macro_rules! unwind {
    ($a:pat, $b:pat) => {
        ($b, $a)
    };
    ($b:pat $(, $list:pat)+) => {
        (unwind!( $( $list ),+ ), $b)
    };
}

#[macro_export]
macro_rules! just {
    ($pattern:pat $(if $guard:expr)? $(,)? => $result:expr) => {
        $crate::parser::SingleParser::new(
            |t| matches!(t, $crate::token::Annotated { token: $pattern, .. } $(if $guard)?),
            |t| {
                match t {
                    $crate::token::Annotated { token: $pattern, .. } $(if $guard)? => Ok($result),
                    t => Err(t)
                }
            })
    };
    ($pattern:pat $(if $guard:expr)? $(,)?) => {
        $crate::parser::SingleParser::new(
            |t| matches!(t, $crate::token::Annotated { token: $pattern, .. } $(if $guard)?),
            |t| {
                match t {
                    $crate::token::Annotated { token: $pattern, .. } $(if $guard)? => Ok(()),
                    t => Err(t)
                }
            })
    };
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{parsers::list_of, toy::Token};

    #[derive(Debug, PartialEq, Eq)]
    struct FooResult(String);
    enum FooInput {
        T(Token),
        S(String),
    }

    impl TryFrom<Token> for FooInput {
        type Error = ();

        fn try_from(value: Token) -> Result<Self, Self::Error> {
            Ok(Self::T(value))
        }
    }

    fn make_line(t: impl IntoIterator<Item = Token>) -> impl Iterator<Item = Annotated<Token>> {
        t.into_iter().enumerate().map(|(i, t)| t.at(1, i + 1))
    }

    #[test]
    fn test_seq() {
        let mut p = parsers::sequence([Token::If, Token::BraceOpen, Token::BraceClose]);

        let mut tokens = make_line([
            Token::If,
            Token::BraceOpen,
            Token::BraceClose,
            Token::LineEnd,
        ]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, vec![Token::If, Token::BraceOpen, Token::BraceClose]);

        let mut tokens = make_line([Token::If, Token::BraceOpen, Token::Comma]);
        let res = p.run_to_completion(&mut tokens);
        assert_eq!(
            res,
            Err("Failed to parse sequence: unexpected comma at ln 1, col 3, expected closing brace `}`".to_string())
        );
    }

    #[test]
    fn test_then() {
        let mut p = just!(Token::BraceOpen).then(just!(Token::BraceClose));

        let mut tokens = make_line([Token::BraceOpen, Token::BraceClose]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, ((), ()));

        let mut tokens = make_line([Token::BraceOpen, Token::If]);
        let err = p.run_to_completion(&mut tokens).err().unwrap();
        assert_eq!(
            err,
            "Failed to parse single token: unexpected keyword `if` at ln 1, col 2".to_string()
        )
    }

    #[test]
    fn test_delimited() {
        let mut p = parsers::discard_delimited([Token::BraceOpen, Token::BraceClose]);

        let mut tokens = make_line([Token::BraceOpen, Token::BraceClose]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, None);

        let mut tokens = make_line([
            Token::BraceOpen,
            Token::Identifier("something".to_string()),
            Token::BraceClose,
        ]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, None);

        let mut tokens = make_line([
            Token::Identifier("something".to_string()),
            Token::BraceClose,
        ]);
        let res = p.run_to_exhaustion(&mut tokens);
        assert_eq!(
            res,
            Err(
                "Failed to parse delimited expression: unexpected token `something` at ln 1, col 1"
                    .to_string()
            )
        );

        let mut tokens = make_line([Token::BraceOpen]);
        let res = p.run_to_exhaustion(&mut tokens);
        assert_eq!(
            res,
            Err("Unexpected end of input: parsing interupted".to_string())
        );
    }

    #[test]
    fn test_list_parser() {
        let var_parser = just!(Token::Identifier(w) => w);
        let mut parser = ListParser::new(Token::Comma, var_parser);

        let mut tokens = make_line([
            Token::Identifier("foo".to_string()),
            Token::Comma,
            Token::Identifier("bar".to_string()),
        ]);
        let res = parser.run_to_completion(&mut tokens);

        assert!(res.is_ok());
        assert_eq!(res.unwrap(), vec!["foo".to_string(), "bar".to_string()]);
    }

    #[test]
    fn test_list_parser_empty() {
        let var_parser = just!(Token::Identifier(w) => w);
        let mut parser = just!(Token::ParenOpen)
            .then(list_of(Token::Comma, var_parser))
            .then(just!(Token::ParenClose))
            .map(|unwind!(_, args, _)| args);

        let mut tokens = make_line([Token::ParenOpen, Token::ParenClose]);
        let res = parser.run_to_completion(&mut tokens);

        assert!(dbg!(&res).is_ok());
        assert!(res.unwrap().is_empty());
    }

    #[test]
    fn test_list_parser_error() {
        let var_parser = just!(Token::Identifier(w) => w);
        let mut parser = just!(Token::ParenOpen)
            .then(list_of(Token::Comma, var_parser))
            .then(just!(Token::ParenClose))
            .map(|unwind!(_, args, _)| args);

        let mut tokens = make_line([
            Token::ParenOpen,
            Token::Identifier("a".to_string()),
            Token::Comma,
            Token::ParenClose,
        ]);

        let res = parser.run_to_completion(&mut tokens);
        assert_eq!(
            res,
            Err(
                "Expected list element: unexpected closing parenthesis `)` at ln 1, col 4"
                    .to_string()
            )
        );
    }

    #[test]
    fn foo() {
        let args = ((1, 2), 3);
        let unwind!(a, b, c) = args;
        assert_eq!(c, 1);
        assert_eq!(b, 2);
        assert_eq!(a, 3);
    }
}
