#![allow(dead_code)]
use std::{
    cell::{RefCell, RefMut},
    fmt::Display,
    marker::PhantomData,
    ops::DerefMut,
    vec,
};

use super::core::{BoxedParser, ParseResult, Parser, PeekResult};
use super::token::Annotated;

pub mod parsers {

    use crate::core::BoxedParser;

    use super::*;

    pub fn lazy<P: Parser + 'static>(p: impl Fn() -> P + 'static) -> LazyParser<P> {
        LazyParser {
            parser: RefCell::new(None),
            get: Box::new(p),
        }
    }

    pub fn one_of<'a, T, E>(parsers: impl IntoIterator<Item = BoxedParser<'a, T, E>>) -> OneOfParser<'a, T, E> {
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
}

pub struct LazyParser<P>
where
    P: Parser,
{
    parser: RefCell<Option<P>>,
    get: Box<dyn Fn() -> P>,
}

impl<P> LazyParser<P>
where
    P: Parser,
{
    fn get_parser(&self) -> impl DerefMut<Target = P> + '_ {
        let initialized = {
            let cur = self.parser.borrow();
            cur.is_some()
        };
        if !initialized {
            let new_parser = (self.get)();
            self.parser.borrow_mut().replace(new_parser);
        }
        RefMut::map(self.parser.borrow_mut(), |p| p.as_mut().unwrap())
    }
}

impl<P> Parser for LazyParser<P>
where
    P: Parser,
    P::Token: Display,
{
    type Expression = P::Expression;
    type Token = P::Token;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        self.get_parser().peek(token)
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        self.get_parser().parse(token, next_token)
    }

    fn reset(&mut self) {
        *self.parser.borrow_mut() = None;
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

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        let Self { seq, i, .. } = self;
        let Some(cur) = seq.get(*i) else {
            return PeekResult::WouldFail("Expected to parse at least one token".to_string());
        };
        let expecting_more = seq.get(*i + 1).is_some();
        let success = &token.token == cur;
        match (success, expecting_more) {
            (true, false) => PeekResult::WouldComplete,
            (true, true) => PeekResult::WouldAccept,
            (false, _) => PeekResult::WouldFail(format!("Failed to parse sequence")),
        }
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let Self { seq, i, collected } = self;
        let Some(cur) = seq.get(*i) else {
            return ParseResult::Failed("Expected to parse at least one token".to_string(), token);
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
                    token,
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

impl<T: Display, E> Parser for SingleParser<T, E> {
    type Expression = E;
    type Token = T;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        match (self.token_matches)(&token) {
            true => PeekResult::WouldComplete,
            false => PeekResult::WouldFail("Failed to parse single token".to_string()),
        }
    }

    fn parse(
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

    fn reset(&mut self) {}
}

pub struct CompletingParser<T, E>(PhantomData<T>, Box<dyn Fn() -> E>);

impl<T: Display, E> Parser for CompletingParser<T, E> {
    type Expression = E;
    type Token = T;

    fn parse(
        &mut self,
        _: Annotated<Self::Token>,
        _: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        ParseResult::Complete(self.1(), None)
    }

    fn reset(&mut self) {}

    fn peek(&self, _: &Annotated<Self::Token>) -> PeekResult {
        PeekResult::WouldComplete
    }
}

pub struct CandidateParser<'a, T, E> {
    is_candidate: bool,
    parser: BoxedParser<'a, T, E>,
}
pub struct OneOfParser<'a, T, E>(Vec<CandidateParser<'a, T, E>>);

impl<'a, T, E> OneOfParser<'a, T, E> {
    pub fn new(parsers: Vec<BoxedParser<'a, T, E>>) -> Self {
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

impl<'a, T, E> Parser for OneOfParser<'a, T, E>
where
    T: Display + Clone,
{
    type Expression = E;
    type Token = T;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        self.0.iter().filter(|p| p.is_candidate).fold(
            PeekResult::WouldFail("No expression matched".to_string()),
            |res, p| match res {
                PeekResult::WouldFail(_) => p.parser.as_ref().peek(token),
                s => s,
            },
        )
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let result = self.0.iter_mut().filter(|p| p.is_candidate).fold(
            ParseResult::Failed("No expression matched".to_string(), token),
            |res, p| match res {
                // Candidate not found
                ParseResult::Failed(_, failed_token) => {
                    let result = p.parser.as_mut().parse(failed_token, next_token);
                    if !matches!(result, ParseResult::Accepted(_)) {
                        p.is_candidate = false;
                    }
                    result
                }
                // Candidate already found
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
            it.parser.as_mut().reset();
        });
    }
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

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        match self.parser.peek(token) {
            PeekResult::WouldComplete => PeekResult::WouldAccept,
            s => s,
        }
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let next_is_end = next_token.map(|t| t.token == self.end).unwrap_or(false);
        match self.parser.parse(token, next_token) {
            ParseResult::Accepted(t) => {
                self.busy = true;
                ParseResult::Accepted(t)
            }
            ParseResult::Complete(e, t) if next_is_end => {
                self.acc.push(e);
                let r = ParseResult::Complete(self.acc.drain(..).collect(), t);
                self.reset();
                r
            }
            ParseResult::Complete(e, t) => {
                self.busy = false;
                self.acc.push(e);
                ParseResult::Accepted(t)
            }
            ParseResult::Failed(e, t) => ParseResult::Failed(e, t),
        }
    }

    fn reset(&mut self) {
        self.busy = false;
        self.acc = vec![];
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

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        let is_separator = token.token == self.separator;
        match is_separator {
            true => PeekResult::WouldAccept,
            false => match self.item_parser.peek(token) {
                PeekResult::WouldFail(_) if self.acc.is_empty() => PeekResult::WouldComplete,
                _ => PeekResult::WouldAccept,
            },
        }
    }

    fn parse(
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
            (false, _) => match self.item_parser.parse(token, next_token) {
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
                    ParseResult::Complete(vec![], Some(t))
                }
                ParseResult::Failed(_, t) => self.fail_token("Expected list element", t),
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
        $crate::combinators::SingleParser::new(
            |t| matches!(t, $crate::token::Annotated { token: $pattern, .. } $(if $guard)?),
            |t| {
                match t {
                    $crate::token::Annotated { token: $pattern, .. } $(if $guard)? => Ok($result),
                    t => Err(t)
                }
            })
    };
    ($pattern:pat $(if $guard:expr)? $(,)?) => {
        $crate::combinators::SingleParser::new(
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
            res.map_err(|err| err.message),
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
            err.message,
            "Failed to parse single token: unexpected keyword `if` at ln 1, col 2: unexpected opening brace `{` at ln 1, col 1".to_string()
        )
    }

    #[test]
    fn test_list_parser() {
        let var_parser = just!(Token::Identifier(_w) => _w);
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
        let var_parser = just!(Token::Identifier(_w) => _w);
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
        let var_parser = just!(Token::Identifier(_w) => _w);
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
            res.map_err(|err| err.message),
            Err("Expected list element: unexpected closing parenthesis `)` at ln 1, col 4".to_string())
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
