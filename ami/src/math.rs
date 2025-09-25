use std::fmt::Display;

use crate::{
    core::{ParseResult, Parser, PeekResult},
    token::Annotated,
};

#[derive(PartialEq, Eq)]
pub enum OpKind {
    Binary,
    Prefix,
    Postfix,
    GroupStart,
    GroupEnd,
}

pub enum Assoc {
    Left,
    Right,
}

pub struct OpToken<T> {
    pub token: T,
    precedence: u8,
    associativity: Option<Assoc>,
    pub kind: OpKind,
}

impl<T> OpToken<T> {
    pub fn new_binary(t: T, precedence: u8, associativity: Assoc) -> Self {
        return OpToken {
            token: t,
            precedence,
            associativity: Some(associativity),
            kind: OpKind::Binary,
        };
    }

    pub fn new_prefix(t: T, precedence: u8) -> Self {
        return OpToken {
            token: t,
            precedence,
            associativity: None,
            kind: OpKind::Prefix,
        };
    }

    pub fn new_postfix(t: T, precedence: u8) -> Self {
        return OpToken {
            token: t,
            precedence,
            associativity: None,
            kind: OpKind::Postfix,
        };
    }

    pub fn new_start_group(t: T) -> Self {
        return OpToken {
            token: t,
            precedence: 0,
            associativity: None,
            kind: OpKind::GroupStart,
        };
    }
    pub fn new_end_group(t: T) -> Self {
        return OpToken {
            token: t,
            precedence: 0,
            associativity: None,
            kind: OpKind::GroupEnd,
        };
    }

    pub fn has_priority(&self, other: &Self) -> bool {
        self.precedence > other.precedence
            || self.precedence == other.precedence
                && matches!(self.associativity, Some(Assoc::Right))
    }
}

pub trait MathExpression: Sized {
    type Token;

    fn as_operator(token: &Self::Token, kind: OpKind) -> Option<OpToken<Self::Token>>;

    fn as_operand(token: Annotated<Self::Token>) -> Result<Self, Annotated<Self::Token>>;

    fn combine(lhs: Option<Self>, op: OpToken<Self::Token>, rhs: Option<Self>) -> Self;
}

enum MathParserState<T, E> {
    Initial,
    LHSParsed(E),
    OperatorParsed(Option<E>, OpToken<T>),
    ParsingNested(
        Option<E>,
        Option<OpToken<T>>,
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

            MathParserState::LHSParsed(lhs) => self.parse_main_operator(lhs, token, next),

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
                        Some(op) => {
                            self.maybe_complete(E::combine(lhs, op, Some(rhs)), token, next)
                        }
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
                    [OpKind::Binary, OpKind::Postfix, OpKind::GroupEnd],
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

    fn as_operator<const N: usize>(&self, token: &T, kinds: [OpKind; N]) -> Option<OpToken<T>> {
        kinds
            .into_iter()
            .fold(None, |op, kind| op.or_else(|| E::as_operator(token, kind)))
    }

    fn parse_operators<const N: usize>(
        &mut self,
        token: Annotated<T>,
        kinds: [OpKind; N],
    ) -> ParseResult<T, OpToken<T>> {
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

    fn parse_main_operator(
        &mut self,
        lhs: E,
        token: Annotated<T>,
        next: Option<&Annotated<T>>,
    ) -> ParseResult<T, E> {
        self.parse_operators(token, [OpKind::Binary, OpKind::Postfix, OpKind::GroupEnd])
            .flat_map(|op, t| match op.kind {
                OpKind::Binary => {
                    self.state = MathParserState::OperatorParsed(Some(lhs), op);
                    ParseResult::Accepted(t)
                }
                OpKind::Postfix => {
                    let final_expr = E::combine(Some(lhs), op, None);
                    self.maybe_complete(final_expr, t, next)
                }
                OpKind::GroupEnd => self.complete(lhs, t),
                _ => unreachable!(),
            })
    }

    fn parse_non_operand(
        &mut self,
        current_lhs: Option<E>,
        current_op: Option<OpToken<T>>,
        token: Annotated<T>,
    ) -> ParseResult<T, E> {
        self.parse_operators(token, [OpKind::Prefix, OpKind::GroupStart])
            .flat_map(|next_op, t| match next_op.kind {
                OpKind::Prefix => match current_op {
                    Some(current_op) => {
                        let next_parser = Box::new(Self {
                            state: MathParserState::OperatorParsed(None, next_op),
                            min_precedence: current_op.precedence,
                        });
                        self.state = MathParserState::ParsingNested(
                            current_lhs,
                            Some(current_op),
                            next_parser,
                        );

                        ParseResult::Accepted(t)
                    }
                    None => {
                        self.state = MathParserState::OperatorParsed(None, next_op);
                        ParseResult::Accepted(t)
                    }
                },
                OpKind::GroupStart => {
                    let next_parser = Box::new(Self {
                        state: MathParserState::Initial,
                        min_precedence: 0,
                    });
                    self.state =
                        MathParserState::ParsingNested(current_lhs, current_op, next_parser);
                    ParseResult::Accepted(t)
                }
                _ => unreachable!(),
            })
    }

    fn parse_rhs(
        &mut self,
        current_lhs: Option<E>,
        current_op: OpToken<T>,
        rhs: E,
        next_token: Option<&Annotated<T>>,
    ) -> ParseResult<T, E> {
        let next_op = next_token.and_then(|t| {
            self.as_operator(
                &t.token,
                [OpKind::Binary, OpKind::Postfix, OpKind::GroupEnd],
            )
        });

        match next_op {
            Some(next_op) if next_op.has_priority(&current_op) => {
                let next_parser = Box::new(Self {
                    state: MathParserState::LHSParsed(rhs),
                    min_precedence: current_op.precedence,
                });
                self.state =
                    MathParserState::ParsingNested(current_lhs, Some(current_op), next_parser);
                ParseResult::Accepted(None)
            }
            Some(next_op)
                if next_op.kind == OpKind::GroupEnd
                    || next_op.precedence <= current_op.precedence
                        && next_op.precedence > self.min_precedence =>
            {
                let next_lhs = E::combine(current_lhs, current_op, Some(rhs));
                self.state = MathParserState::LHSParsed(next_lhs);
                ParseResult::Accepted(None)
            }
            _ => {
                let final_expr = E::combine(current_lhs, current_op, Some(rhs));
                self.complete(final_expr, None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        token::Numeric64,
        toy::{self, Token},
    };

    fn make_line(t: impl IntoIterator<Item = Token>) -> impl Iterator<Item = Annotated<Token>> {
        t.into_iter().enumerate().map(|(i, t)| t.at(1, i + 1))
    }

    fn int(i: i64) -> Token {
        Token::LitNum(Numeric64::Int(i))
    }

    #[test]
    fn test_math() {
        let mut parser = MathExpressionParser::<toy::Token, ToyMath>::new();

        // -0 + 1 + 1 x 0 + -1
        let mut tokens = make_line([
            Token::Minus,
            int(0),
            Token::Plus,
            int(1),
            Token::Plus,
            int(1),
            Token::Times,
            int(0),
            Token::Plus,
            Token::Minus,
            int(1),
        ]);

        let res = parser.run_to_completion(&mut tokens);
        assert_eq!("(((-(0) + 1) + (1 * 0)) + -(1))", res.unwrap().to_string());
    }

    #[test]
    fn test_math2() {
        let mut parser = MathExpressionParser::<toy::Token, ToyMath>::new();

        // 1 x (0 + 1) x 0 ^ 1 x 2!
        let mut tokens = make_line([
            int(1),
            Token::Times,
            Token::ParenOpen,
            int(0),
            Token::Plus,
            int(1),
            Token::ParenClose,
            Token::Times,
            int(0),
            Token::Exponent,
            int(1),
            Token::Times,
            int(2),
            Token::Factorial,
        ]);

        let res = parser.run_to_completion(&mut tokens);
        assert_eq!("(((1 * (0 + 1)) * (0^1)) * (2)!)", res.unwrap().to_string());
    }

    #[test]
    fn test_math3() {
        let mut parser = MathExpressionParser::<toy::Token, ToyMath>::new();

        let mut tokens = make_line([int(1)]);

        let res = parser.run_to_completion(&mut tokens);
        assert_eq!("1", res.unwrap().to_string());
    }

    #[derive(Debug)]
    enum ToyMath {
        Atom(i64),
        Add(Box<ToyMath>, Box<ToyMath>),
        Mul(Box<ToyMath>, Box<ToyMath>),
        Exp(Box<ToyMath>, Box<ToyMath>),
        Neg(Box<ToyMath>),
        Fact(Box<ToyMath>),
    }

    impl Display for ToyMath {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ToyMath::Atom(b) => f.write_str(&b.to_string()),
                ToyMath::Add(lhs, rhs) => f.write_fmt(format_args!("({lhs} + {rhs})")),
                ToyMath::Mul(lhs, rhs) => f.write_fmt(format_args!("({lhs} * {rhs})")),
                ToyMath::Neg(rhs) => f.write_fmt(format_args!("-({rhs})")),
                ToyMath::Exp(lhs, rhs) => f.write_fmt(format_args!("({lhs}^{rhs})")),
                ToyMath::Fact(lhs) => f.write_fmt(format_args!("({lhs})!")),
            }
        }
    }

    impl MathExpression for ToyMath {
        type Token = toy::Token;

        fn as_operator(token: &Token, kind: OpKind) -> Option<OpToken<Token>> {
            match (token, kind) {
                (Token::Times, OpKind::Binary) => {
                    Some(OpToken::new_binary(Token::Times, 2, Assoc::Left))
                }
                (Token::Plus, OpKind::Binary) => {
                    Some(OpToken::new_binary(Token::Plus, 1, Assoc::Left))
                }
                (Token::Exponent, OpKind::Binary) => {
                    Some(OpToken::new_binary(Token::Exponent, 2, Assoc::Right))
                }
                (Token::Minus, OpKind::Prefix) => Some(OpToken::new_prefix(Token::Minus, 1)),
                (Token::Factorial, OpKind::Postfix) => {
                    Some(OpToken::new_postfix(Token::Factorial, 3))
                }
                (Token::ParenOpen, OpKind::GroupStart) => {
                    Some(OpToken::new_start_group(Token::ParenOpen))
                }
                (Token::ParenClose, OpKind::GroupEnd) => {
                    Some(OpToken::new_end_group(Token::ParenClose))
                }
                _ => None,
            }
        }

        fn as_operand(token: Annotated<Token>) -> Result<Self, Annotated<Token>> {
            match &token.token {
                toy::Token::LitNum(Numeric64::Int(i)) => Ok(ToyMath::Atom(*i)),
                _ => Err(token),
            }
        }

        fn combine(lhs: Option<Self>, op: OpToken<Token>, rhs: Option<Self>) -> Self {
            match (lhs, op.kind, op.token, rhs) {
                (Some(lhs), OpKind::Binary, Token::Times, Some(rhs)) => {
                    Self::Mul(Box::new(lhs), Box::new(rhs))
                }
                (Some(lhs), OpKind::Binary, Token::Exponent, Some(rhs)) => {
                    Self::Exp(Box::new(lhs), Box::new(rhs))
                }
                (Some(lhs), OpKind::Binary, Token::Plus, Some(rhs)) => {
                    Self::Add(Box::new(lhs), Box::new(rhs))
                }
                (None, OpKind::Prefix, Token::Minus, Some(rhs)) => Self::Neg(Box::new(rhs)),
                (Some(lhs), OpKind::Postfix, Token::Factorial, None) => Self::Fact(Box::new(lhs)),
                _ => unreachable!(),
            }
        }
    }
}
