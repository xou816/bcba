use ami::parser::{Parser, SingleParser};
use ami::parsers::*;
use ami::token::{Annotated, Tokenizer};
use ami::{expect_token, unwind};
use ami::toy::Token;


#[derive(Debug)]
enum Statement {
    IfStatement(bool, Box<Statement>),
    CallProc(String),
}

impl Statement {
    fn exec(&self) {
        match self {
            Statement::IfStatement(cond, stm) if *cond => stm.exec(),
            Statement::CallProc(proc) => match proc.as_str() {
                "hello" => {
                    println!("Hello world!");
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn parser() -> impl Parser<Expression = Self, Token = Token> {
        one_of([
            expect_token!(Token::If)
                .then(expect_token!(t @ (Token::False|Token::True) => t == Token::True))
                .then(expect_token!(Token::BraceOpen))
                .then_lazy(Statement::parser)
                .then(expect_token!(Token::BraceClose))
                .map(|unwind!(_, stm, _, cond, _)| {
                    Self::IfStatement(cond, Box::new(stm))
                })
                .boxed(),
            expect_token!(Token::Word(w) => w)
                .then(expect_token!(Token::ParenOpen))
                .then(expect_token!(Token::ParenClose))
                .map(|unwind!(_, _, proc)| Self::CallProc(proc))
                .boxed(),
        ])
    }
}

fn main() {
    let mut tokens = Tokenizer::tokenize::<Token>("if true { hello() }");
    let mut p = Statement::parser();
    let res = p.run_to_completion(&mut tokens).unwrap();
    res.exec();
}

