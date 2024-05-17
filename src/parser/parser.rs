#![allow(dead_code)]
use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    vec,
};

use super::token::Annotated;

#[derive(Debug)]
pub enum ParseResult<Token, Result> {
    Accepted,
    Ignored(Annotated<Token>),
    Complete(Result),
    Failed(String),
}

impl<T, Result> ParseResult<T, Result> {
    fn map_result<F, R2>(self, f: F) -> ParseResult<T, R2>
    where
        F: FnOnce(Result) -> R2,
    {
        match self {
            ParseResult::Accepted => ParseResult::Accepted,
            ParseResult::Ignored(t) => ParseResult::Ignored(t),
            ParseResult::Complete(r) => ParseResult::Complete(f(r)),
            ParseResult::Failed(e) => ParseResult::Failed(e),
        }
    }
}

pub trait Parser {
    type Expression;
    type Token: Display;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression>;

    fn consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        match self.try_consume(token, next_token) {
            ParseResult::Ignored(t) => self.fail_token(t),
            r => r,
        }
    }

    fn run_to_completion(
        &mut self,
        tokens: &mut dyn Iterator<Item = Annotated<Self::Token>>,
    ) -> Result<Self::Expression, String> {
        let mut tokens = tokens.peekable();
        loop {
            let Some(token) = tokens.next() else {
                break Err("Unexpected end of input".to_string());
            };
            let next = tokens.peek();
            match self.consume(token, next) {
                ParseResult::Complete(res) => break Ok(res),
                ParseResult::Failed(err) => break Err(err),
                ParseResult::Accepted => continue,
                ParseResult::Ignored(_) => panic!(),
            }
        }
    }

    fn run_to_exhaustion(
        &mut self,
        tokens: &mut dyn Iterator<Item = Annotated<Self::Token>>,
    ) -> Result<Vec<Self::Expression>, String> {
        let mut tokens = tokens.peekable();
        let mut acc: Vec<Self::Expression> = vec![];
        loop {
            let Some(token) = tokens.next() else {
                break Ok(acc);
            };
            let next = tokens.peek();
            match self.consume(token, next) {
                ParseResult::Complete(res) => acc.push(res),
                ParseResult::Failed(err) => break Err(err),
                ParseResult::Accepted if next.is_none() => {
                    break (Err("Unexpected end of input".to_string()))
                }
                ParseResult::Accepted => continue,
                ParseResult::Ignored(_) => panic!(),
            }
        }
    }

    fn reset(&mut self);

    fn complete(&mut self, r: Self::Expression) -> ParseResult<Self::Token, Self::Expression> {
        self.reset();
        ParseResult::Complete(r)
    }

    fn fail_token(
        &mut self,
        token: Annotated<Self::Token>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        self.reset();
        ParseResult::Failed(format!(
            "Unexpected {} at ln {}, col {}",
            token.token, token.row, token.col
        ))
    }

    fn fail(&mut self, e: String) -> ParseResult<Self::Token, Self::Expression> {
        self.reset();
        ParseResult::Failed(e)
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
                .consume(token, next_token)
                .map_result(|b| (self.cur_result.take().unwrap(), b))
        } else {
            match self.cur.try_consume(token, next_token) {
                ParseResult::Accepted => ParseResult::Accepted,
                ParseResult::Ignored(t) => ParseResult::Ignored(t),
                ParseResult::Complete(a) => {
                    self.cur_result = Some(a);
                    ParseResult::Accepted
                }
                ParseResult::Failed(e) => ParseResult::Failed(e),
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

    pub fn list_of<T, E>(
        separator: T,
        item_parser: impl Parser<Token = T, Expression = E> + 'static,
    ) -> ListParser<T, E> {
        ListParser::new(separator, item_parser)
    }

    pub fn discard_delimited<T>(delimiter: [T; 2]) -> DelimitedParser<T, ()>
    where
        T: Display + Clone + Eq + 'static,
    {
        let end = delimiter[1].clone();
        DelimitedParser {
            delimiter,
            inner_result: None,
            inner_parser: DiscardUntil(end).boxed(),
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
            return ParseResult::Failed("Expected to parse at least one token".to_string());
        };
        let next = seq.get(*i + 1);
        let success = &token.token == cur && next_token.map(|t| &t.token) == next || next == None;
        match (success, next) {
            (true, None) => {
                collected.push(token.token);
                let r = ParseResult::Complete(collected.drain(..).collect());
                self.reset();
                r
            }
            (true, Some(_)) => {
                *i += 1;
                collected.push(token.token);
                ParseResult::Accepted
            }
            (false, _) => {
                if *i == 0 {
                    ParseResult::Ignored(token)
                } else {
                    self.fail_token(token)
                }
            }
        }
    }

    fn reset(&mut self) {
        self.i = 0;
        self.collected = vec![];
    }
}

pub struct SingleParser<T, E> {
    token_matches: Box<dyn Fn(Annotated<T>, Option<&Annotated<T>>) -> Result<E, Annotated<T>>>,
}

impl<T, E> SingleParser<T, E> {
    pub fn new(
        token_matches: impl Fn(Annotated<T>, Option<&Annotated<T>>) -> Result<E, Annotated<T>> + 'static,
    ) -> Self {
        Self {
            token_matches: Box::new(token_matches),
        }
    }
}

impl<T: Display, E> Parser for SingleParser<T, E> {
    type Expression = E;
    type Token = T;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let token_matches = &self.token_matches;
        match token_matches(token, next_token) {
            Ok(r) => self.complete(r),
            Err(t) => ParseResult::Ignored(t),
        }
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
        ParseResult::Complete(self.1())
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
            ParseResult::Complete(r) => match self.1(r) {
                Ok(r) => self.complete(r),
                Err(msg) => self.fail(msg),
            },
            ParseResult::Ignored(r) => ParseResult::Ignored(r),
            ParseResult::Accepted => ParseResult::Accepted,
            ParseResult::Failed(e) => self.fail(e),
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
    T: Display,
{
    type Expression = E;
    type Token = T;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let result = self.0.iter_mut().filter(|p| p.is_candidate).fold(
            ParseResult::Ignored(token),
            |res, p| match res {
                ParseResult::Ignored(token) => match p.parser.try_consume(token, next_token) {
                    r @ ParseResult::Accepted => r,
                    r => {
                        p.is_candidate = false;
                        r
                    }
                },
                _ => {
                    p.is_candidate = false;
                    res
                }
            },
        );
        if matches!(result, ParseResult::Complete(_) | ParseResult::Failed(_)) {
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
            Some(Annotated { token, .. }) if token == &self.0 => ParseResult::Complete(()),
            _ => ParseResult::Accepted,
        }
    }

    fn reset(&mut self) {}
}

pub struct DelimitedParser<T, E> {
    delimiter: [T; 2],
    inner_result: Option<E>,
    inner_parser: BoxedParser<T, E>,
    started: bool,
}

impl<T, E> Parser for DelimitedParser<T, E>
where
    T: Display + Eq,
    E: Default,
{
    type Expression = E;
    type Token = T;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        let [ref start, ref end] = self.delimiter;
        let inner_done = self.inner_result.is_some();
        match (
            &token.token == start,
            &token.token == end,
            self.started,
            inner_done,
        ) {
            (true, _, false, _) => {
                self.started = true;
                ParseResult::Accepted
            }
            (false, false, true, false) => match self.inner_parser.consume(token, next_token) {
                ParseResult::Complete(r) => {
                    self.inner_result = Some(r);
                    ParseResult::Accepted
                }
                r => r,
            },
            (_, true, _, _) => {
                let result = ParseResult::Complete(self.inner_result.take().unwrap_or_default());
                self.reset();
                result
            }
            (false, _, false, _) => ParseResult::Ignored(token),
            _ => self.fail_token(token),
        }
    }

    fn reset(&mut self) {
        self.inner_result = None;
        self.inner_parser.reset();
        self.started = false;
    }
}

pub struct ListParser<T, E> {
    separator: T,
    item_parser: BoxedParser<T, E>,
    acc: Vec<E>,
}

impl<T, E> ListParser<T, E> {
    fn new(separator: T, item_parser: impl Parser<Token = T, Expression = E> + 'static) -> Self {
        Self {
            separator,
            item_parser: item_parser.boxed(),
            acc: vec![],
        }
    }
}

impl<T: Display, E> Parser for ListParser<T, E>
where
    T: Eq,
{
    type Expression = Vec<E>;
    type Token = T;

    fn try_consume(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        if token.token == self.separator {
            return ParseResult::Accepted;
        }

        match self.item_parser.try_consume(token, next_token) {
            ParseResult::Accepted => ParseResult::Accepted,
            ParseResult::Complete(item) => {
                self.acc.push(item);
                if next_token.is_none()
                    || matches!(next_token, Some(Annotated {token,..}) if token != &self.separator)
                {
                    ParseResult::Complete(self.acc.drain(..).collect())
                } else {
                    ParseResult::Accepted
                }
            }
            ParseResult::Failed(e) => self.fail(e),
            ParseResult::Ignored(t) if self.acc.is_empty() => ParseResult::Ignored(t),
            ParseResult::Ignored(t) => self.fail_token(t),
        }
    }

    fn reset(&mut self) {
        self.acc = vec![];
    }
}

macro_rules! unwind {
    ($a:pat, $b:pat) => {
        ($b, $a)
    };
    ($b:pat $(, $list:pat)+) => {
        (unwind!( $( $list ),+ ), $b)
    };
}

macro_rules! expect_token {
    ($pattern:pat $(if $guard:expr)? $(,)? => $result:expr) => {
        SingleParser::new(|t, _| {
            match t {
                Annotated { token: $pattern, .. } $(if $guard)? => Ok($result),
                t => Err(t)
            }
        })
    };
    ($pattern:pat $(if $guard:expr)? $(,)?) => {
        SingleParser::new(|t, _| {
            match t {
                Annotated { token: $pattern, .. } $(if $guard)? => Ok(()),
                t => Err(t)
            }
        })
    };
}

pub(crate) use expect_token;
pub(crate) use unwind;

#[cfg(test)]
mod tests {

    use tests::parsers::discard_delimited;

    use super::*;
    use crate::parser::token::{Token, TokenDeserialize};

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
        let mut p = parsers::sequence([Token::CommentStart, Token::CommentEnd])
            .then(expect_token!(Token::LineEnd));

        let mut tokens = make_line([Token::CommentStart, Token::CommentEnd, Token::LineEnd]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, (vec![Token::CommentStart, Token::CommentEnd], ()));
    }

    #[test]
    fn test_then() {
        let mut p = expect_token!(Token::CommentStart).then(expect_token!(Token::CommentEnd));

        let mut tokens = make_line([Token::CommentStart, Token::CommentEnd]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, ((), ()));

        let mut tokens = make_line([Token::CommentStart, Token::LedgerEntryStart]);
        let err = p.run_to_completion(&mut tokens).err().unwrap();
        assert_eq!(err, "Unexpected start of ledger at ln 1, col 2".to_string())
    }

    #[test]
    fn test_delimited() {
        let mut p = discard_delimited([Token::CommentStart, Token::CommentEnd]);

        let mut tokens = make_line([Token::CommentStart, Token::CommentEnd]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, ());

        let mut tokens = make_line([
            Token::CommentStart,
            Token::Word("something".to_string()),
            Token::CommentEnd,
        ]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, ());

        let mut tokens = make_line([Token::Word("something".to_string()), Token::CommentEnd]);
        let res = p.run_to_exhaustion(&mut tokens);
        assert_eq!(res, Err("Unexpected token `something` at ln 1, col 1".to_string()));

        let mut tokens = make_line([Token::CommentStart]);
        let res = p.run_to_exhaustion(&mut tokens);
        assert_eq!(res, Err("Unexpected end of input".to_string()));
    }

    #[test]
    fn test_list_parser() {
        let name_tag_parser = expect_token!(Token::NameAnchor)
            .then(expect_token!(Token::Word(w) => w))
            .map(|(_, name)| name);

        let mut parser = ListParser::new(Token::Comma, name_tag_parser);
        let mut tokens = make_line([
            Token::NameAnchor,
            Token::Word("Alex".to_string()),
            Token::Comma,
            Token::NameAnchor,
            Token::Word("Toto".to_string()),
        ]);
        let res = parser.run_to_completion(&mut tokens);

        assert!(res.is_ok());
        assert_eq!(res.unwrap(), vec!["Alex".to_string(), "Toto".to_string()])
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
