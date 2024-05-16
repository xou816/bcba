#![allow(dead_code)]
use std::{
    fmt::{Debug, Display},
    iter::Peekable,
    vec,
};

use super::{
    token::{AnnotatedToken, Token},
    Expression,
};

#[derive(Debug)]
pub enum ParseResult<Result> {
    Accepted,
    Ignored(AnnotatedToken),
    Complete(Result),
    Failed(String),
}

impl<Result> ParseResult<Result> {
    fn map_result<F, R2>(self, f: F) -> ParseResult<R2>
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

pub trait IntoParser {
    fn parse() -> BoxedParser<Self>;
}

pub trait Parser {
    type Expression;
    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression>;

    fn consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
        match self.try_consume(token, next_token) {
            ParseResult::Ignored(t) => self.fail_token(t),
            r => r,
        }
    }

    fn run_to_completion(
        &mut self,
        tokens: &mut dyn Iterator<Item = AnnotatedToken>,
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
        tokens: &mut dyn Iterator<Item = AnnotatedToken>,
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

    fn complete(&mut self, r: Self::Expression) -> ParseResult<Self::Expression> {
        self.reset();
        ParseResult::Complete(r)
    }

    fn fail_token(&mut self, token: AnnotatedToken) -> ParseResult<Self::Expression> {
        self.reset();
        ParseResult::Failed(format!(
            "Unexpected {} at ln {}, col {}",
            token.token.display_name(),
            token.row,
            token.col
        ))
    }

    fn fail(&mut self, e: String) -> ParseResult<Self::Expression> {
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

    fn boxed(self) -> Box<dyn Parser<Expression = Self::Expression>>
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
    B: Parser,
{
    type Expression = (A::Expression, B::Expression);

    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
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
    use serde_json::map::{IntoIter, Iter};

    use super::*;

    pub fn one_of<T: 'static>(parsers: Vec<BoxedParser<T>>) -> OneOfParser<T> {
        OneOfParser::new(parsers)
    }

    pub fn always_completing<T: 'static, F>(result_factory: F) -> CompletingParser<T>
    where
        F: Fn() -> T + 'static,
    {
        CompletingParser(Box::new(result_factory))
    }

    pub fn expect(
        seq: impl IntoIterator<Item = Token, IntoIter = impl Iterator<Item = Token>>,
    ) -> SeqParser {
        SeqParser {
            seq: seq.into_iter().collect(),
            i: 0,
            collected: vec![],
        }
    }

    pub fn list_of<E>(
        separator: Token,
        item_parser: impl Parser<Expression = E> + 'static,
    ) -> ListParser<E> {
        ListParser::new(separator, item_parser)
    }

    pub fn expect_delimited(t: [Token; 2]) -> DelimitedParser {
        DelimitedParser::new(t)
    }
}

pub struct SeqParser {
    seq: Vec<Token>,
    i: usize,
    collected: Vec<Token>,
}

impl Parser for SeqParser {
    type Expression = Vec<Token>;

    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
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
            (false, _) => self.fail_token(token),
        }
    }

    fn reset(&mut self) {
        self.i = 0;
        self.collected = vec![];
    }
}

pub struct SingleParser<E> {
    token_matches: Box<dyn Fn(Token, Option<&Token>) -> Result<E, String>>,
}

impl<E> SingleParser<E> {
    pub fn new(token_matches: impl Fn(Token, Option<&Token>) -> Result<E, String> + 'static) -> Self {
        Self {
            token_matches: Box::new(token_matches),
        }
    }
}

impl<E> Parser for SingleParser<E> {
    type Expression = E;

    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
        let token_matches = &self.token_matches;
        match token_matches(token.token, next_token.map(|t| &t.token)) {
            Ok(r) => self.complete(r),
            Err(e) => self.fail(e),
        }
    }

    fn reset(&mut self) {}
}

pub struct CompletingParser<T>(Box<dyn Fn() -> T>);

impl<T> Parser for CompletingParser<T> {
    type Expression = T;

    fn try_consume(
        &mut self,
        _: AnnotatedToken,
        _: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
        ParseResult::Complete(self.0())
    }

    fn reset(&mut self) {}
}

pub struct SuffixedParser<P>
where
    P: Parser + Sized,
{
    child: P,
    token: Token,
    result: Option<P::Expression>,
}

impl<P> SuffixedParser<P>
where
    P: Parser + Sized,
{
    fn new(child: P, token: Token) -> Self {
        Self {
            child,
            token,
            result: None,
        }
    }
}

impl<P> Parser for SuffixedParser<P>
where
    P: Parser + Sized,
{
    type Expression = P::Expression;

    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
        if self.result.is_some() {
            if token.token == self.token {
                self.child.reset();
                ParseResult::Complete(self.result.take().unwrap())
            } else {
                self.fail(format!(
                    "Unexpected {} at ln {}, col {}, expected {}",
                    token.token.display_name(),
                    token.row,
                    token.col,
                    self.token.display_name()
                ))
            }
        } else {
            match self.child.try_consume(token, next_token) {
                ParseResult::Complete(r) => {
                    self.result.replace(r);
                    ParseResult::Accepted
                }
                ParseResult::Failed(e) => self.fail(e),
                r => r,
            }
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.result = None;
    }
}

pub struct PrefixedParser<P>
where
    P: Parser + Sized,
{
    child: P,
    token: Token,
    prefix_parsed: bool,
}

impl<P> PrefixedParser<P>
where
    P: Parser + Sized,
{
    fn new(child: P, token: Token) -> Self {
        Self {
            child,
            token,
            prefix_parsed: false,
        }
    }
}

impl<P> Parser for PrefixedParser<P>
where
    P: Parser + Sized,
{
    type Expression = P::Expression;

    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
        if !self.prefix_parsed {
            if token.token == self.token {
                self.prefix_parsed = true;
                ParseResult::Accepted
            } else {
                ParseResult::Ignored(token)
            }
        } else {
            match self.child.consume(token, next_token) {
                ParseResult::Complete(r) => self.complete(r),
                ParseResult::Failed(e) => self.fail(e),
                r => r,
            }
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.prefix_parsed = false;
    }
}

pub type BoxedParser<T> = Box<dyn Parser<Expression = T>>;

impl<T> Parser for BoxedParser<T> {
    type Expression = T;

    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
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

    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
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

pub struct CandidateParser<T> {
    is_candidate: bool,
    parser: BoxedParser<T>,
}
pub struct OneOfParser<T>(Vec<CandidateParser<T>>);

impl<T: 'static> OneOfParser<T> {
    fn new(parsers: Vec<BoxedParser<T>>) -> Self {
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

impl<T: 'static> Parser for OneOfParser<T> {
    type Expression = T;

    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
        let result = self.0.iter_mut().filter(|p| p.is_candidate).fold(
            ParseResult::Ignored(token),
            |res, p| match res {
                ParseResult::Ignored(token) => {
                    match p.parser.try_consume(token.clone(), next_token) {
                        r @ ParseResult::Accepted => r,
                        ParseResult::Failed(_) => {
                            p.is_candidate = false;
                            ParseResult::Ignored(token)
                        }
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

pub struct DelimitedParser {
    delimiter: [Token; 2],
    acc: Vec<Token>,
    depth: usize,
}

impl DelimitedParser {
    fn new(delimiter: [Token; 2]) -> Self {
        Self {
            delimiter,
            acc: vec![],
            depth: 0,
        }
    }
}

impl Parser for DelimitedParser {
    type Expression = Vec<Token>;

    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        _: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
        let [ref start, ref end] = self.delimiter;
        let depth = self.depth;
        match token.token {
            ref t if t == start => {
                self.depth += 1;
                ParseResult::Accepted
            }
            ref t if t == end => match depth {
                0 => self.fail_token(token),
                1 => {
                    let result = self.acc.drain(..).collect();
                    self.complete(result)
                }
                _ => {
                    self.depth -= 1;
                    ParseResult::Accepted
                }
            },
            _ => {
                if depth > 0 {
                    self.acc.push(token.token);
                    ParseResult::Accepted
                } else {
                    self.fail_token(token)
                }
            }
        }
    }

    fn reset(&mut self) {
        self.acc = vec![];
        self.depth = 0;
    }
}

pub struct ListParser<E> {
    separator: Token,
    item_parser: BoxedParser<E>,
    acc: Vec<E>,
}

impl<E> ListParser<E> {
    fn new(separator: Token, item_parser: impl Parser<Expression = E> + 'static) -> Self {
        Self {
            separator,
            item_parser: item_parser.boxed(),
            acc: vec![],
        }
    }
}

impl<E> Parser for ListParser<E> {
    type Expression = Vec<E>;

    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
        if token.token == self.separator {
            return ParseResult::Accepted;
        }

        match self.item_parser.consume(token, next_token) {
            ParseResult::Accepted => ParseResult::Accepted,
            ParseResult::Complete(item) => {
                self.acc.push(item);
                if next_token.is_none()
                    || matches!(next_token, Some(AnnotatedToken {token,..}) if token != &self.separator)
                {
                    ParseResult::Complete(self.acc.drain(..).collect())
                } else {
                    ParseResult::Accepted
                }
            }
            ParseResult::Failed(e) => ParseResult::Failed(e),
            ParseResult::Ignored(_) => unreachable!(),
        }
    }

    fn reset(&mut self) {
        self.acc = vec![];
    }
}

pub type ResultCombiner<Input, Expression> = Box<dyn Fn(Vec<Input>) -> Result<Expression, String>>;

enum SequenceParserStep<I> {
    Expect(Token),
    Save(Box<dyn Fn(&Token) -> bool>),
    SaveUntil(Box<dyn Fn(&Token) -> bool>),
    Delegate(BoxedParser<I>),
}

impl<I> Display for SequenceParserStep<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SequenceParserStep::Expect(token) => {
                f.write_fmt(format_args!("Expect {}", token.display_name()))
            }
            SequenceParserStep::Save(_) => f.write_str("Save tokens"),
            SequenceParserStep::SaveUntil(_) => f.write_str("Save until ??"),
            SequenceParserStep::Delegate(_) => f.write_str("Delegate to other parser"),
        }
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
                $pattern $(if $guard)? => Ok($result),
                _ => Err(format!("Unexpected {}", t.display_name()))
            }
        })
    };
    ($pattern:pat $(if $guard:expr)? $(,)?) => {
        SingleParser::new(|t, _| {
            match &t {
                $pattern $(if $guard)? => Ok(t),
                _ => Err(format!("Unexpected {}", t.display_name()))
            }
        })
    };
}

pub(crate) use expect_token;
pub(crate) use unwind;

#[cfg(test)]
mod tests {

    use super::*;

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

    fn make_line(t: impl IntoIterator<Item = Token>) -> impl Iterator<Item = AnnotatedToken> {
        t.into_iter().enumerate().map(|(i, t)| t.at(1, i + 1))
    }

    #[test]
    fn test_seq() {
        let mut p = parsers::expect([Token::CommentStart, Token::CommentEnd])
            .then(expect_token!(Token::LineEnd));

        let mut tokens = make_line([Token::CommentStart, Token::CommentEnd, Token::LineEnd]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(
            res,
            (vec![Token::CommentStart, Token::CommentEnd], Token::LineEnd)
        );
    }

    #[test]
    fn test_then() {
        let mut p = expect_token!(Token::CommentStart).then(expect_token!(Token::CommentEnd));

        let mut tokens = make_line([Token::CommentStart, Token::CommentEnd]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, (Token::CommentStart, Token::CommentEnd));

        let mut tokens = make_line([Token::CommentStart, Token::LedgerEntryStart]);
        let err = p.run_to_completion(&mut tokens).err().unwrap();
        assert_eq!(err, "Unexpected start of ledger".to_string())
    }

    #[test]
    fn test_delimited() {
        let mut p = DelimitedParser::new([Token::CommentStart, Token::CommentEnd]);

        let mut tokens = make_line([Token::CommentStart, Token::CommentEnd]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, vec![]);

        let mut tokens = make_line([
            Token::CommentStart,
            Token::Word("something".to_string()),
            Token::CommentEnd,
        ]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, vec![Token::Word("something".to_string())]);

        let mut tokens = make_line([
            Token::CommentStart,
            Token::CommentStart,
            Token::Word("something".to_string()),
            Token::CommentEnd,
            Token::CommentEnd,
        ]);
        let res = p.run_to_completion(&mut tokens).unwrap();
        assert_eq!(res, vec![Token::Word("something".to_string())]);

        let mut tokens = make_line([Token::Comma]);
        let res = p.run_to_completion(&mut tokens);
        assert_eq!(res, Err("Unexpected comma at ln 1, col 1".to_string()));

        let mut tokens = make_line([
            Token::CommentStart,
            Token::CommentStart,
            Token::Word("something".to_string()),
            Token::CommentEnd,
        ]);
        let res = p.run_to_exhaustion(&mut tokens);
        assert_eq!(res, Err("Unexpected end of input".to_string()));

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
