use std::fmt::Display;

use super::token::Annotated;

pub enum PeekResult {
    WouldAccept,
    WouldComplete,
    WouldFail(String),
}

#[derive(Debug)]
pub enum ParseResult<Token, Result> {
    Accepted(Option<Annotated<Token>>),
    Complete(Result, Option<Annotated<Token>>),
    Failed(String, Annotated<Token>),
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


pub trait Parser {
    type Expression;
    type Token: Display;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult;

    fn parse(
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
            match self.parse(token, peeked_token) {
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
            match self.parse(token, next) {
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

    fn reset(&mut self);

    fn complete(&mut self, r: Self::Expression) -> ParseResult<Self::Token, Self::Expression> {
        self.reset();
        ParseResult::Complete(r, None)
    }

    fn fail(
        &mut self,
        message: String,
        token: Annotated<Self::Token>,
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
            token,
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
        P: Parser<Token = Self::Token>,
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
    B: Parser<Token = A::Token>,
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

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        match self.cur_result {
            Some(_) => self.next.peek(token),
            None => match self.cur.peek(token) {
                PeekResult::WouldComplete => PeekResult::WouldAccept,
                s => s,
            },
        }
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        if self.cur_result.is_some() {
            self.next
                .parse(token, next_token)
                .map_result(|b| (self.cur_result.take().unwrap(), b))
        } else {
            let peek_error = match self.cur.peek(&token) {
                PeekResult::WouldComplete => match next_token.map(|n| self.next.peek(n)) {
                    Some(PeekResult::WouldFail(e)) => {
                        let desc = next_token.unwrap().describe();
                        Some(format!("{e}: unexpected {desc}"))
                    }
                    _ => None,
                },
                PeekResult::WouldFail(e) => Some(e),
                _ => None,
            };
            if let Some(peek_error) = peek_error {
                return self.fail(peek_error, token);
            }
            match self.cur.parse(token, next_token) {
                ParseResult::Accepted(t) => ParseResult::Accepted(t),
                ParseResult::Complete(a, t) => {
                    self.cur_result = Some(a);
                    ParseResult::Accepted(t)
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

pub type BoxedParser<T, E> = Box<dyn Parser<Token = T, Expression = E>>;

impl<T: Display, E> Parser for BoxedParser<T, E> {
    type Expression = E;
    type Token = T;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        self.as_ref().peek(token)
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        self.as_mut().parse(token, next_token)
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

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        self.0.peek(token)
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        next_token: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        match self.0.parse(token, next_token) {
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
