use ami::prelude::*;
use ami::parsers::*;
use ami::token::{Annotated, Tokenizer};
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
            just!(Token::If)
                .then(just!(t @ (Token::False|Token::True) => t == Token::True))
                .then(just!(Token::BraceOpen))
                .then_lazy(Statement::parser)
                .then(just!(Token::BraceClose))
                .map(|unwind!(_, stm, _, cond, _)| {
                    Self::IfStatement(cond, Box::new(stm))
                })
                .boxed(),
            just!(Token::Word(w) => w)
                .then(sequence([Token::ParenOpen, Token::ParenClose]))
                .map(|unwind!(_, proc)| Self::CallProc(proc))
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

