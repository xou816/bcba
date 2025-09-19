use std::fmt::Display;

use crate::{
    core::{ParseResult, Parser, PeekResult},
    token::Annotated,
};

#[derive(PartialEq, Eq)]
pub enum OperatorKind {
    Binary,
    Unary,
    PrecedenceGroupStart,
    PrecedenceGroupEnd,
}

pub struct OperatorToken<T> {
    pub token: T,
    precedence: u8,
    pub kind: OperatorKind,
}

impl<T> OperatorToken<T> {
    pub fn new_binary(t: T, precedence: u8) -> Self {
        return OperatorToken {
            token: t,
            precedence,
            kind: OperatorKind::Binary,
        };
    }

    pub fn new_unary(t: T, precedence: u8) -> Self {
        return OperatorToken {
            token: t,
            precedence,
            kind: OperatorKind::Unary,
        };
    }

    pub fn new_start_group(t: T) -> Self {
        return OperatorToken {
            token: t,
            precedence: 0,
            kind: OperatorKind::PrecedenceGroupStart,
        };
    }
    pub fn new_end_group(t: T) -> Self {
        return OperatorToken {
            token: t,
            precedence: 0,
            kind: OperatorKind::PrecedenceGroupEnd,
        };
    }
}

pub trait MathExpression: Sized {
    type Token;

    fn as_operator(token: &Self::Token, kind: OperatorKind) -> Option<OperatorToken<Self::Token>>;

    fn as_operand(token: Annotated<Self::Token>) -> Result<Self, Annotated<Self::Token>>;

    fn combine(lhs: Option<Self>, op: OperatorToken<Self::Token>, rhs: Self) -> Self;
}

enum MathParserState<T, E> {
    Initial,
    LHSParsed(E),
    OperatorParsed(Option<E>, OperatorToken<T>),
    ParsingNested(
        Option<E>,
        Option<OperatorToken<T>>,
        Box<MathExpressionParser<T, E>>,
    ),
}

pub struct MathExpressionParser<T, E> {
    state: MathParserState<T, E>,
    min_precedence: u8,
}

impl<T, E> Parser for MathExpressionParser<T, E>
where
    T: Display,
    E: MathExpression<Token = T>,
{
    type Token = T;
    type Expression = E;

    fn peek(&self, token: &Annotated<Self::Token>) -> PeekResult {
        PeekResult::WouldAccept
    }

    fn parse(
        &mut self,
        token: Annotated<Self::Token>,
        next: Option<&Annotated<Self::Token>>,
    ) -> ParseResult<Self::Token, Self::Expression> {
        match std::mem::replace(&mut self.state, MathParserState::Initial) {
            MathParserState::Initial => self
                .parse_operand(token)
                .flat_map(|expr, t| self.maybe_complete(expr, t, next))
                .or_else(|t| self.parse_non_operand(None, None, t)),

            MathParserState::LHSParsed(lhs) => self.parse_main_operator(lhs, token),

            MathParserState::OperatorParsed(lhs, op) => match self.parse_operand(token) {
                ParseResult::Complete(expr, _) => self.parse_rhs(lhs, op, expr, next),
                ParseResult::Failed(_, t) => self.parse_non_operand(lhs, Some(op), t),
                ParseResult::Accepted(_) => unreachable!(),
            },

            MathParserState::ParsingNested(lhs, op, mut parser) => {
                match parser.parse(token, next) {
                    ParseResult::Accepted(token) => {
                        self.state = MathParserState::ParsingNested(lhs, op, parser);
                        ParseResult::Accepted(token)
                    }
                    ParseResult::Complete(rhs, token) => match op {
                        Some(op) => self.maybe_complete(E::combine(lhs, op, rhs), token, next),
                        None => self.maybe_complete(rhs, token, next),
                    },
                    ParseResult::Failed(_, token) => self.fail_token("error parsing rhs", token),
                }
            }
        }
    }

    fn reset(&mut self) {
        self.state = MathParserState::Initial;
        self.min_precedence = 0;
    }
}

impl<T, E> MathExpressionParser<T, E>
where
    E: MathExpression<Token = T>,
    T: Display,
{
    pub fn new() -> Self {
        Self {
            state: MathParserState::Initial,
            min_precedence: 0,
        }
    }

    fn maybe_complete(
        &mut self,
        expr: E,
        token: Option<Annotated<T>>,
        next_token: Option<&Annotated<T>>,
    ) -> ParseResult<T, E> {
        if next_token
            .and_then(|t| {
                self.as_operator(
                    &t.token,
                    [OperatorKind::Binary, OperatorKind::PrecedenceGroupEnd],
                )
            })
            .is_some()
        {
            self.state = MathParserState::LHSParsed(expr);
            ParseResult::Accepted(token)
        } else {
            self.complete(expr, token)
        }
    }

    fn as_operator<const N: usize>(
        &self,
        token: &T,
        kinds: [OperatorKind; N],
    ) -> Option<OperatorToken<T>> {
        kinds
            .into_iter()
            .fold(None, |op, kind| op.or_else(|| E::as_operator(token, kind)))
    }

    fn parse_operators<const N: usize>(
        &mut self,
        token: Annotated<T>,
        kinds: [OperatorKind; N],
    ) -> ParseResult<T, OperatorToken<T>> {
        match self.as_operator(&token.token, kinds) {
            Some(op) => ParseResult::Complete(op, None),
            None => ParseResult::Failed("expected operator".to_string(), token),
        }
    }

    fn parse_operand(&mut self, token: Annotated<T>) -> ParseResult<T, E> {
        match E::as_operand(token) {
            Ok(expr) => ParseResult::Complete(expr, None),
            Err(token) => ParseResult::Failed("expected operand".to_string(), token),
        }
    }

    fn parse_main_operator(&mut self, lhs: E, token: Annotated<T>) -> ParseResult<T, E> {
        self.parse_operators(
            token,
            [OperatorKind::Binary, OperatorKind::PrecedenceGroupEnd],
        )
        .flat_map(|op, t| match op.kind {
            OperatorKind::Binary => {
                self.state = MathParserState::OperatorParsed(Some(lhs), op);
                ParseResult::Accepted(t)
            }
            OperatorKind::PrecedenceGroupEnd => self.complete(lhs, t),
            _ => unreachable!(),
        })
    }

    fn parse_non_operand(
        &mut self,
        current_lhs: Option<E>,
        current_op: Option<OperatorToken<T>>,
        token: Annotated<T>,
    ) -> ParseResult<T, E> {
        self.parse_operators(
            token,
            [OperatorKind::Unary, OperatorKind::PrecedenceGroupStart],
        )
        .flat_map(|next_op, t| match next_op.kind {
            OperatorKind::Unary => {
                if let Some(current_op) = current_op {
                    let next_parser = Box::new(Self {
                        state: MathParserState::OperatorParsed(None, next_op),
                        min_precedence: current_op.precedence,
                    });
                    self.state =
                        MathParserState::ParsingNested(current_lhs, Some(current_op), next_parser);
                } else {
                    self.state = MathParserState::OperatorParsed(None, next_op);
                }
                ParseResult::Accepted(t)
            }
            OperatorKind::PrecedenceGroupStart => {
                let next_parser = Box::new(Self {
                    state: MathParserState::Initial,
                    min_precedence: 0,
                });
                self.state = MathParserState::ParsingNested(current_lhs, current_op, next_parser);
                ParseResult::Accepted(t)
            }
            _ => unreachable!(),
        })
    }

    fn parse_rhs(
        &mut self,
        current_lhs: Option<E>,
        current_op: OperatorToken<T>,
        rhs: E,
        next_token: Option<&Annotated<T>>,
    ) -> ParseResult<T, E> {
        let next_op = next_token.and_then(|t| {
            self.as_operator(
                &t.token,
                [OperatorKind::Binary, OperatorKind::PrecedenceGroupEnd],
            )
        });

        match next_op {
            Some(next_op)
                if next_op.kind == OperatorKind::PrecedenceGroupEnd
                    || next_op.precedence <= current_op.precedence
                        && next_op.precedence > self.min_precedence =>
            {
                let next_lhs = E::combine(current_lhs, current_op, rhs);
                self.state = MathParserState::LHSParsed(next_lhs);
                ParseResult::Accepted(None)
            }
            Some(next_op) if next_op.precedence > current_op.precedence => {
                let next_parser = Box::new(Self {
                    state: MathParserState::LHSParsed(rhs),
                    min_precedence: current_op.precedence,
                });
                self.state =
                    MathParserState::ParsingNested(current_lhs, Some(current_op), next_parser);
                ParseResult::Accepted(None)
            }
            _ => {
                let final_expr = E::combine(current_lhs, current_op, rhs);
                self.complete(final_expr, None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toy::{self, Token};

    fn make_line(t: impl IntoIterator<Item = Token>) -> impl Iterator<Item = Annotated<Token>> {
        t.into_iter().enumerate().map(|(i, t)| t.at(1, i + 1))
    }

    #[test]
    fn test_math() {
        let mut parser = MathExpressionParser::<toy::Token, ToyBinaryExp>::new();

        // !false || true || true && false || !true
        let mut tokens = make_line([
            Token::Not,
            Token::False,
            Token::Or,
            Token::True,
            Token::Or,
            Token::True,
            Token::And,
            Token::False,
            Token::Or,
            Token::Not,
            Token::True,
        ]);

        let res = parser.run_to_completion(&mut tokens);
        assert_eq!(
            "(((!(false) || true) || (true && false)) || !(true))",
            res.unwrap().to_string()
        );
    }

    #[test]
    fn test_math2() {
        let mut parser = MathExpressionParser::<toy::Token, ToyBinaryExp>::new();

        // true && (false || true) && false
        let mut tokens = make_line([
            Token::True,
            Token::And,
            Token::ParenOpen,
            Token::False,
            Token::Or,
            Token::True,
            Token::ParenClose,
            Token::And,
            Token::False,
        ]);

        let res = parser.run_to_completion(&mut tokens);
        assert_eq!(
            "((true && (false || true)) && false)",
            res.unwrap().to_string()
        );
    }

    #[test]
    fn test_math3() {
        let mut parser = MathExpressionParser::<toy::Token, ToyBinaryExp>::new();

        let mut tokens = make_line([Token::True]);

        let res = parser.run_to_completion(&mut tokens);
        assert_eq!("true", res.unwrap().to_string());
    }

    #[derive(Debug)]
    enum ToyBinaryExp {
        Atom(bool),
        Or(Box<ToyBinaryExp>, Box<ToyBinaryExp>),
        And(Box<ToyBinaryExp>, Box<ToyBinaryExp>),
        Not(Box<ToyBinaryExp>),
    }

    impl Display for ToyBinaryExp {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ToyBinaryExp::Atom(b) => f.write_str(&b.to_string()),
                ToyBinaryExp::Or(lhs, rhs) => f.write_fmt(format_args!("({lhs} || {rhs})")),
                ToyBinaryExp::And(lhs, rhs) => f.write_fmt(format_args!("({lhs} && {rhs})")),
                ToyBinaryExp::Not(rhs) => f.write_fmt(format_args!("!({rhs})")),
            }
        }
    }

    impl MathExpression for ToyBinaryExp {
        type Token = toy::Token;

        fn as_operator(
            token: &Self::Token,
            kind: OperatorKind,
        ) -> Option<OperatorToken<Self::Token>> {
            match (token, kind) {
                (Self::Token::And, OperatorKind::Binary) => {
                    Some(OperatorToken::new_binary(Self::Token::And, 2))
                }
                (Self::Token::Or, OperatorKind::Binary) => {
                    Some(OperatorToken::new_binary(Self::Token::Or, 1))
                }
                (Self::Token::Not, OperatorKind::Unary) => {
                    Some(OperatorToken::new_unary(Self::Token::Not, 1))
                }
                (Self::Token::ParenOpen, OperatorKind::PrecedenceGroupStart) => {
                    Some(OperatorToken::new_start_group(Self::Token::ParenOpen))
                }
                (Self::Token::ParenClose, OperatorKind::PrecedenceGroupEnd) => {
                    Some(OperatorToken::new_end_group(Self::Token::ParenClose))
                }
                _ => None,
            }
        }

        fn as_operand(token: Annotated<Self::Token>) -> Result<Self, Annotated<Self::Token>> {
            match &token.token {
                toy::Token::True => Ok(ToyBinaryExp::Atom(true)),
                toy::Token::False => Ok(ToyBinaryExp::Atom(false)),
                _ => Err(token),
            }
        }

        fn combine(lhs: Option<Self>, op: OperatorToken<Self::Token>, rhs: Self) -> Self {
            match (lhs, op.kind, op.token) {
                (Some(lhs), OperatorKind::Binary, Self::Token::And) => {
                    Self::And(Box::new(lhs), Box::new(rhs))
                }
                (Some(lhs), OperatorKind::Binary, Self::Token::Or) => {
                    Self::Or(Box::new(lhs), Box::new(rhs))
                }
                (None, OperatorKind::Unary, Self::Token::Not) => Self::Not(Box::new(rhs)),
                _ => unreachable!(),
            }
        }
    }
}
