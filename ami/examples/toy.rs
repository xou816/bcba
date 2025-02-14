use std::collections::HashMap;

use ami::parsers::*;
use ami::prelude::*;
use ami::token::Tokenizer;
use ami::toy::Token;

#[derive(Debug, Clone)]
enum Value {
    Str(String),
    Bool(bool),
}

impl Value {
    fn str_coerce(&self) -> &str {
        match self {
            Value::Bool(true) => "true",
            Value::Bool(false) => "false",
            Value::Str(s) => s,
        }
    }

    fn bool_coerce(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Str(s) => !s.is_empty(),
        }
    }
}

#[derive(Debug)]
enum BoolAtom {
    Lit(bool),
    Ref(String),
}

impl BoolAtom {
    fn parser() -> impl Parser<Token = Token, Expression = Self> {
        one_of([
            just!(t @ (Token::False | Token::True) => Self::Lit(t == Token::True)).boxed(),
            just!(Token::Identifier(s) => Self::Ref(s)).boxed(),
        ])
    }
}

#[derive(Debug)]
enum BoolExpression {
    Atom(BoolAtom),
    And(BoolAtom, Box<BoolExpression>),
}

impl BoolExpression {
    fn parser() -> impl Parser<Token = Token, Expression = Self> {
        one_of([
            BoolAtom::parser()
                .then(just!(Token::And))
                .then(lazy(Self::parser))
                .map(|unwind!(ex, _, a)| Self::And(a, Box::new(ex)))
                .boxed(),
            BoolAtom::parser().map(|b| Self::Atom(b)).boxed(),
        ])
    }
}

#[derive(Debug)]
enum Expression {
    Lit(Value),
    Ref(String),
    // BoolExpression(BoolExpression)
}

impl Expression {
    fn eval<'a>(&'a self, context: &'a HashMap<String, Value>) -> &'a Value {
        match self {
            Expression::Lit(value) => value,
            Expression::Ref(k) => context.get(k).expect("unresolved var"),
        }
    }

    fn parser() -> impl Parser<Token = Token, Expression = Self> {
        one_of([
            just!(Token::LitString(s) => Self::Lit(Value::Str(s))).boxed(),
            just!(t @ (Token::False | Token::True) => Self::Lit(Value::Bool(t == Token::True)))
                .boxed(),
            just!(Token::Identifier(s) => Self::Ref(s)).boxed(),
        ])
    }
}

#[derive(Debug)]
enum Statement {
    IfStatement(Expression, Vec<Statement>),
    AssignStatement(String, Expression),
    CallFunc(String, Vec<Expression>),
    Empty,
}

impl Statement {
    fn exec(self, context: &mut HashMap<String, Value>) {
        match self {
            Statement::IfStatement(cond, stms) if cond.eval(context).bool_coerce() => {
                for stm in stms {
                    stm.exec(context);
                }
            }
            Statement::CallFunc(func, args) => match func.as_str() {
                "hello" => println!("Hello world!"),
                "bye" => println!("Bye world!"),
                "print" => println!("{}", args[0].eval(context).str_coerce()),
                _ => {}
            },
            Statement::AssignStatement(id, exp) => {
                context.insert(id, exp.eval(context).clone());
            }
            _ => {}
        }
    }

    fn parser() -> impl Parser<Expression = Self, Token = Token> {
        one_of([
            just!(Token::If)
                .then(Expression::parser())
                .then(just!(Token::BraceOpen))
                .then(repeat_until(Token::BraceClose, lazy(Statement::parser)))
                .then(just!(Token::BraceClose))
                .map(|unwind!(_, stm, _, cond, _)| Self::IfStatement(cond, stm))
                .boxed(),
            just!(Token::Identifier(w) => w)
                .then(just!(Token::ParenOpen))
                .then(list_of(Token::Comma, Expression::parser()))
                .then(just!(Token::ParenClose))
                .map(|unwind!(_, args, _, func)| Self::CallFunc(func, args))
                .boxed(),
            just!(Token::Let)
                .then(just!(Token::Identifier(w) => w))
                .then(just!(Token::Assign))
                .then(Expression::parser())
                .map(|unwind!(exp, _, id, _)| Self::AssignStatement(id, exp))
                .boxed(),
            just!(Token::LineEnd).map(|_| Self::Empty).boxed(),
        ])
    }
}

fn main() {
    let mut tokens = Tokenizer::<Token>::new().tokenize("true && false");
    let mut p = BoolExpression::parser();
    dbg!(p.run_to_completion(&mut tokens).unwrap());

    let mut tokens = Tokenizer::<Token>::new().tokenize(
        r#"
    let fun = true
    let message = "lets party!"
    if polite {
        print("i am polite!")
        if fun {
            print(message)
        }
    }
    "#,
    );

    let mut context = HashMap::new();
    context.insert("polite".to_string(), Value::Bool(true));

    Statement::parser()
        .run_to_exhaustion(&mut tokens)
        .unwrap()
        .into_iter()
        .for_each(|s| {
            s.exec(&mut context);
        });
}
