use ami::parsers::*;
use ami::prelude::*;
use ami::token::Tokenizer;
use ami::toy::Token;

#[derive(Debug)]
enum Statement {
    IfStatement(bool, Vec<Statement>),
    CallProc(String),
    Empty,
}

impl Statement {
    fn exec(&self) {
        match self {
            Statement::IfStatement(cond, stms) if *cond => {
                for stm in stms {
                    stm.exec();
                }
            }
            Statement::CallProc(proc) => match proc.as_str() {
                "hello" => {
                    println!("Hello world!");
                }
                "bye" => {
                    println!("Bye world!");
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
                .then(delimited(
                    [Token::BraceOpen, Token::BraceClose],
                    list_of(Token::LineEnd, lazy(Statement::parser)),
                ))
                .map(|unwind!(stm, cond, _)| match stm {
                    Some(stm) => Self::IfStatement(cond, stm),
                    None => Self::Empty,
                })
                .boxed(),
            just!(Token::Word(w) => w)
                .then(sequence([Token::ParenOpen, Token::ParenClose]))
                .map(|unwind!(_, proc)| Self::CallProc(proc))
                .boxed(),
            just!(Token::LineEnd).map(|_| Self::Empty).boxed(),
        ])
    }
}

fn main() {
    let mut tokens = Tokenizer::tokenize::<Token>(
        r#"
    if true {
        hello()
    }

    hello()
    bye()
    "#,
    );
    let mut p = Statement::parser();
    p.run_to_exhaustion(&mut tokens)
        .unwrap()
        .into_iter()
        .for_each(|s| {
            s.exec();
        });
}
