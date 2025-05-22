use std::collections::HashMap;

use ami::parsers::*;
use ami::prelude::*;
use ami::toy::toy_tokenizer;
use ami::toy::Token;
use itertools::Itertools;

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

    fn and(self, other: Self) -> Self {
        Value::Bool(self.bool_coerce() && other.bool_coerce())
    }
}

#[derive(Debug)]
enum BoolAtom {
    Lit(bool),
    Ref(String),
}

impl BoolAtom {
    fn eval(&self, context: &HashMap<String, Value>) -> Value {
        match self {
            Self::Lit(b) => Value::Bool(*b),
            Self::Ref(k) => context
                .get(k)
                .expect(&format!("cannot read value of {k}"))
                .to_owned(),
        }
    }

    fn parser() -> impl Parser<Token = Token, Expression = Self> {
        one_of([
            just!(_t @ (Token::False | Token::True) => Self::Lit(_t == Token::True)).boxed(),
            just!(Token::Identifier(_s) => Self::Ref(_s)).boxed(),
        ])
    }
}

#[derive(Debug)]
enum BoolExpression {
    Atom(BoolAtom),
    And(BoolAtom, Box<BoolExpression>),
}

impl BoolExpression {
    fn eval(&self, context: &HashMap<String, Value>) -> Value {
        match self {
            Self::Atom(a) => a.eval(context),
            Self::And(a, ex) => a.eval(context).and(ex.eval(&context)),
        }
    }

    fn parser() -> impl Parser<Token = Token, Expression = Self> {
        one_of([
            BoolAtom::parser()
                .then(just!(Token::And))
                .then(lazy(Self::parser))
                .map(|unwind!(ex, _, a)| Self::And(a, Box::new(ex)))
                .boxed(),
            BoolAtom::parser().map(Self::Atom).boxed(),
        ])
    }
}

#[derive(Debug)]
enum Expression {
    Lit(Value),
    Ref(String),
    BoolExpression(BoolExpression),
}

impl Expression {
    fn eval(&self, context: &HashMap<String, Value>) -> Value {
        match self {
            Self::Lit(value) => value.clone(),
            Self::Ref(k) => context.get(k).expect("unresolved var").clone(),
            Self::BoolExpression(ex) => ex.eval(context),
        }
    }

    fn parser() -> impl Parser<Token = Token, Expression = Self> {
        one_of([
            BoolExpression::parser().map(Self::BoolExpression).boxed(),
            just!(Token::LitString(_s) => Self::Lit(Value::Str(_s))).boxed(),
            just!(Token::Identifier(_s) => Self::Ref(_s)).boxed(),
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
                "print" => println!(
                    "{}",
                    args.iter()
                        .map(|a| a.eval(context).str_coerce().to_owned())
                        .collect::<String>()
                ),
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
            just!(Token::Identifier(_w) => _w)
                .then(just!(Token::ParenOpen))
                .then(list_of(Token::Comma, Expression::parser()))
                .then(just!(Token::ParenClose))
                .map(|unwind!(_, args, _, func)| Self::CallFunc(func, args))
                .boxed(),
            just!(Token::Let)
                .then(just!(Token::Identifier(_w) => _w))
                .then(just!(Token::Assign))
                .then(Expression::parser())
                .map(|unwind!(exp, _, id, _)| Self::AssignStatement(id, exp))
                .boxed(),
            just!(Token::LineEnd).map(|_| Self::Empty).boxed(),
        ])
    }
}

fn main() {
    let program = r#"
    let polite = true
    let fun = true
    let message = "lets party!"
    if polite {
        print("i am polite!")
        if fun {
            print(message)
        }
    }
    print("true && false: ", true && false)
    "#;

    let mut tokens = toy_tokenizer().tokenize(program);

    let mut context = HashMap::new();

    let res = Statement::parser().run_to_exhaustion(&mut tokens);

    match res {
        Ok(st) => st.into_iter().for_each(|s| {
            s.exec(&mut context);
        }),
        Err(e) => {
            let ctx = toy_tokenizer()
                .tokenize(program)
                .skip_while(|t| t.row < e.row)
                .take_while(|t| t.col < e.col)
                .map(|t| t.token.render())
                .join(" ");
            println!("{ctx}\n{}^ {}", " ".repeat(ctx.len()), e.message)
        }
    }
}
