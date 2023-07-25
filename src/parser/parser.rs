#![allow(dead_code)]
use std::fmt::{Display, Debug};

use super::token::{AnnotatedToken, Token};

#[derive(Debug)]
pub enum ParseResult<Result> {
    Accepted,
    Ignored(AnnotatedToken),
    Complete(Result),
    Failed(String),
}

pub trait Parser {
    type Expression;
    fn try_consume(&mut self, token: AnnotatedToken, next_token: Option<&AnnotatedToken>) -> ParseResult<Self::Expression>;

    fn consume(&mut self, token: AnnotatedToken, next_token: Option<&AnnotatedToken>) -> ParseResult<Self::Expression> {
        match self.try_consume(token, next_token) {
            ParseResult::Ignored(t) => ParseResult::Failed(format!(
                "Unexpected {} at ln {}, col {}",
                t.token.display_name(),
                t.row,
                t.col
            )),
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
                ParseResult::Ignored(_) => panic!()
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
                ParseResult::Accepted if next.is_none() => break(Err("Unexpected end of input".to_string())),
                ParseResult::Accepted => continue,
                ParseResult::Ignored(_) => panic!()
            }
        }
    }

    fn reset(&mut self);

    fn complete(&mut self, r: Self::Expression) -> ParseResult<Self::Expression> {
        self.reset();
        ParseResult::Complete(r)
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
        MapParser(self, Box::new(f))
    }

    fn prefixed(self, token: Token) -> PrefixedParser<Self>
    where
        Self: Sized,
    {
        PrefixedParser::new(self, token)
    }

    fn suffixed(self, token: Token) -> SuffixedParser<Self>
    where
        Self: Sized,
    {
        SuffixedParser::new(self, token)
    }

    fn boxed(self) -> Box<dyn Parser<Expression = Self::Expression>>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

pub struct Parsers;

impl Parsers {
    pub fn one_of<T: 'static>(parsers: Vec<BoxedParser<T>>) -> OneOfParser<T> {
        OneOfParser::new(parsers)
    }

    pub fn always_completing<T: 'static, F>(result_factory: F) -> CompletingParser<T> where F: Fn() -> T + 'static {
        CompletingParser(Box::new(result_factory))
    }
}

pub struct CompletingParser<T>(Box<dyn Fn() -> T>);

impl <T> Parser for CompletingParser<T> {
    type Expression = T;

    fn try_consume(&mut self, _: AnnotatedToken, _: Option<&AnnotatedToken>) -> ParseResult<Self::Expression> {
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

    fn try_consume(&mut self, token: AnnotatedToken, next_token: Option<&AnnotatedToken>) -> ParseResult<Self::Expression> {
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
                },
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

    fn try_consume(&mut self, token: AnnotatedToken, next_token: Option<&AnnotatedToken>) -> ParseResult<Self::Expression> {
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

pub struct MapParser<P, U>(P, Box<dyn Fn(P::Expression) -> U>)
where
    P: Parser + Sized;

impl<P, U> Parser for MapParser<P, U>
where
    P: Parser + Sized,
{
    type Expression = U;

    fn try_consume(&mut self, token: AnnotatedToken, next_token: Option<&AnnotatedToken>) -> ParseResult<Self::Expression> {
        match self.0.try_consume(token, next_token) {
            ParseResult::Complete(r) => self.complete(self.1(r)),
            ParseResult::Ignored(r) => ParseResult::Ignored(r),
            ParseResult::Accepted => ParseResult::Accepted,
            ParseResult::Failed(e) => self.fail(e),
        }
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

pub struct CandidateParser<T>(bool, BoxedParser<T>);
pub struct OneOfParser<T>(Vec<CandidateParser<T>>);

impl<T: 'static> OneOfParser<T> {
    fn new(parsers: Vec<BoxedParser<T>>) -> Self {
        Self(
            parsers
                .into_iter()
                .map(|p| CandidateParser(true, p))
                .collect(),
        )
    }
}

impl<T: 'static> Parser for OneOfParser<T> {
    type Expression = T;

    fn try_consume(&mut self, token: AnnotatedToken, next_token: Option<&AnnotatedToken>) -> ParseResult<Self::Expression> {
        let result = self.0.iter_mut().filter(|p| p.0).fold(
            ParseResult::Ignored(token),
            |res, p| match res {
                ParseResult::Ignored(token) => match p.1.try_consume(token, next_token) {
                    r @ ParseResult::Accepted => r,
                    r => {
                        p.0 = false;
                        r
                    }
                },
                _ => res,
            },
        );
        if matches!(result, ParseResult::Complete(_) | ParseResult::Failed(_)) {
            self.reset();
        }
        result
    }

    fn reset(&mut self) {
        self.0.iter_mut().for_each(|it| {
            it.0 = true;
            it.1.reset();
        });
    }
}

pub struct ListParser<E> {
    separator: Token,
    item_parser: BoxedParser<E>,
    acc: Vec<E>
}

impl <E> ListParser<E> {
    pub fn new(separator: Token, item_parser: impl Parser<Expression = E> + 'static) -> Self {
        Self {
            separator,
            item_parser: item_parser.boxed(),
            acc: vec![]
        }
    }
}

impl <E> Parser for ListParser<E> {
    type Expression = Vec<E>;

    fn try_consume(&mut self, token: AnnotatedToken, next_token: Option<&AnnotatedToken>) -> ParseResult<Self::Expression> {
        if token.token == self.separator {
            return ParseResult::Accepted;
        }

        match self.item_parser.consume(token, next_token) {
            ParseResult::Accepted => ParseResult::Accepted,
            ParseResult::Complete(item) => {
                self.acc.push(item);
                if next_token.is_none() || matches!(next_token, Some(AnnotatedToken {token,..}) if token != &self.separator) {
                    ParseResult::Complete(self.acc.drain(..).collect())
                } else {
                    ParseResult::Accepted
                }
            },
            ParseResult::Failed(e) => ParseResult::Failed(e),
            ParseResult::Ignored(_) => panic!(),
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

impl <I> Display for SequenceParserStep<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SequenceParserStep::Expect(token) => f.write_fmt(format_args!("Expect {}", token.display_name())),
            SequenceParserStep::Save(_) => f.write_str("Save tokens"),
            SequenceParserStep::SaveUntil(_) => f.write_str("Save until ??"),
            SequenceParserStep::Delegate(_) => f.write_str("Delegate to other parser"),
        }
    }
}

pub struct SequenceParser<Result, Expression> {
    steps: Vec<SequenceParserStep<Result>>,
    current_step: usize,
    results: Vec<Result>,
    combiner: ResultCombiner<Result, Expression>,
}
pub struct SequenceParserBuilder<StepResult> {
    steps: Vec<SequenceParserStep<StepResult>>,
}

impl<StepResult> SequenceParserBuilder<StepResult>
where
    StepResult: TryFrom<Token>,
{
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn expect(mut self, t: Token) -> Self {
        self.steps.push(SequenceParserStep::Expect(t));
        self
    }

    pub fn save(mut self, f: impl Fn(&Token) -> bool + 'static) -> Self {
        self.steps.push(SequenceParserStep::Save(Box::new(f)));
        self
    }

    pub fn save_until(mut self, f: impl Fn(&Token) -> bool + 'static) -> Self {
        self.steps.push(SequenceParserStep::SaveUntil(Box::new(f)));
        self
    }


    pub fn delegate(mut self, p: impl Parser<Expression = StepResult> + 'static) -> Self {
        self.steps.push(SequenceParserStep::Delegate(p.boxed()));
        self
    }

    pub fn combine<F, Input, Expression>(self, f: F) -> SequenceParser<StepResult, Expression>
    where
        F: Fn(Vec<StepResult>) -> Result<Expression, String> + 'static,
    {
        SequenceParser {
            steps: self.steps.into(),
            current_step: 0,
            results: vec![],
            combiner: Box::new(f),
        }
    }
}

impl<Result, Expression> SequenceParser<Result, Expression>
where
    Result: TryFrom<Token>,
{
    fn fail_if_needed(&mut self, token: AnnotatedToken) -> ParseResult<Expression> {
        let started = self.current_step > 0;
        if started {
            self.fail(format!(
                "Unexpected {} at ln {}, col {}",
                token.token.display_name(),
                token.row,
                token.col
            ))
        } else {
            ParseResult::Ignored(token)
        }
    }
}

impl<Result, Expression> Parser for SequenceParser<Result, Expression>
where
    Result: TryFrom<Token>,
{
    type Expression = Expression;

    fn try_consume(&mut self, token: AnnotatedToken, next_token: Option<&AnnotatedToken>) -> ParseResult<Self::Expression> {
        let Some(step) = self.steps.get_mut(self.current_step) else {
            return ParseResult::Failed("Invalid state".to_string());
        };
        // println!("{}:{} {}", token.row, token.col, step);
        let result: ParseResult<Expression> = match (step, token) {
            (SequenceParserStep::Expect(expected), AnnotatedToken { token, .. })
                if expected == &token =>
            {
                self.current_step += 1;
                ParseResult::Accepted
            }
            (SequenceParserStep::Expect(expected), token) if expected != &token.token => {
                self.fail_if_needed(token)
            }
            (SequenceParserStep::Save(matcher), token) if matcher(&token.token) => {
                self.current_step += 1;
                if let Ok(t) = token.token.try_into() {
                    self.results.push(t);
                }
                ParseResult::Accepted
            }
            (SequenceParserStep::Save(matcher), token) if !matcher(&token.token) => {
                self.fail_if_needed(token)
            }
            (SequenceParserStep::SaveUntil(matcher), token) if matcher(&token.token) => {
                self.current_step += 1;
                if let Ok(t) = token.token.try_into() {
                    self.results.push(t);
                }
                ParseResult::Accepted
            }
            (SequenceParserStep::SaveUntil(matcher), token) if !matcher(&token.token) => {
                if let Ok(t) = token.token.try_into() {
                    self.results.push(t);
                }
                ParseResult::Accepted
            }
            (SequenceParserStep::Delegate(parser), token) => match parser.consume(token, next_token) {
                ParseResult::Accepted => ParseResult::Accepted,
                ParseResult::Complete(r) => {
                    self.current_step += 1;
                    self.results.push(r);
                    ParseResult::Accepted
                }
                ParseResult::Failed(e) => ParseResult::Failed(e),
                ParseResult::Ignored(_) => panic!("Not allowed to ignore"),
            },
            (_, _) => ParseResult::Failed("Invalid state".to_string()),
        };
        match (self.current_step >= self.steps.len(), result) {
            (true, ParseResult::Accepted) => {
                let f = &self.combiner;
                let expr = f(self.results.drain(..).collect());
                self.complete(expr.unwrap())
            }
            (_, r) => r,
        }
    }

    fn reset(&mut self) {
        self.current_step = 0;
        self.results = vec![];
        self.steps.iter_mut().for_each(|s| {
            if let SequenceParserStep::Delegate(parser) = s {
                parser.reset();
            }
        })
    }
}

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

    #[test]
    fn test_sequence() {
        let name_tag_parser = SequenceParserBuilder::new()
            .expect(Token::NameAnchor)
            .save(|t| matches!(t, Token::Word(_)))
            .combine::<_, Token, String>(|t| {
                let mut t = t.into_iter();
                let Some(Token::Word(w)) = t.next() else { panic!() };
                Ok(w)
            });

        let mut delegating = SequenceParserBuilder::new()
            .expect(Token::CommentStart)
            .delegate(name_tag_parser.map(FooInput::S))
            .expect(Token::CommentEnd)
            .combine::<_, FooInput, FooResult>(|t| {
                let mut t = t.into_iter();
                let Some(FooInput::S(name)) = t.next() else { panic!() };
                Ok(FooResult(name))
            });

        
        let mut tokens = vec![Token::CommentStart, Token::NameAnchor, Token::Word("Alex".to_string()), Token::CommentEnd].into_iter().map(|t| t.at(0, 0));
        let res = delegating.run_to_completion(&mut tokens);

        assert!(res.is_ok());
        assert_eq!(res.unwrap(), FooResult("Alex".to_string()));
    }

    #[test]
    fn test_list_parser() {
        let name_tag_parser = SequenceParserBuilder::new()
        .expect(Token::NameAnchor)
        .save(|t| matches!(t, Token::Word(_)))
        .combine::<_, Token, String>(|t| {
            let mut t = t.into_iter();
            let Some(Token::Word(w)) = t.next() else { panic!() };
            Ok(w)
        });

        let mut parser = ListParser::new(Token::Comma, name_tag_parser);
        let mut tokens = vec![Token::NameAnchor, Token::Word("Alex".to_string()), Token::Comma, Token::NameAnchor, Token::Word("Toto".to_string())].into_iter().map(|t| t.at(0, 0));
        let res = parser.run_to_completion(&mut tokens);

        assert!(res.is_ok());
        assert_eq!(res.unwrap(), vec!["Alex".to_string(), "Toto".to_string()])
    }
}
