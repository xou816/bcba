use std::cmp::Ordering;
use std::collections::HashSet;

use ami::math::Assoc;
use ami::math::MathExpression;
use ami::math::OpKind;
use ami::math::OpToken;
use ami::parsers::*;
use ami::prelude::*;
use ami::token::Annotated;

use super::token::Token;

pub struct LedgerParser;

impl LedgerParser {
    pub fn parse(
        tokens: &mut dyn Iterator<Item = Annotated<Token>>,
    ) -> Result<Vec<Expression>, String> {
        let mut parser = one_of([
            LedgerSection::parser()
                .map(Expression::LedgerSection)
                .tag("ledger")
                .boxed(),
            PersonSection::parser()
                .map(Expression::PersonSection)
                .tag("persons")
                .boxed(),
            just!(Token::LineEnd)
                .map(|_| Expression::None)
                .tag("none")
                .boxed(),
        ])
        .tag("main");
        parser.run_to_exhaustion(tokens).map_err(|e| e.message)
    }
}

#[derive(Debug)]
pub enum Expression {
    PersonSection(PersonSection),
    LedgerSection(LedgerSection),
    None,
}

pub enum ComplexAmount {
    Just(f64),
    Add(Box<ComplexAmount>, Box<ComplexAmount>),
    Sub(Box<ComplexAmount>, Box<ComplexAmount>),
    Mul(Box<ComplexAmount>, Box<ComplexAmount>),
}

impl ComplexAmount {
    fn eval(self) -> f64 {
        match self {
            ComplexAmount::Just(f) => f,
            ComplexAmount::Add(a, b) => a.eval() + b.eval(),
            ComplexAmount::Sub(a, b) => a.eval() - b.eval(),
            ComplexAmount::Mul(a, b) => a.eval() * b.eval(),
        }
    }
}

impl MathExpression for ComplexAmount {
    type Token = Token;

    fn as_operator(token: &Self::Token, kind: OpKind) -> Option<OpToken<Self::Token>> {
        match (token, kind) {
            (Token::Plus, OpKind::Binary) => Some(OpToken::new_binary(Token::Plus, 1, Assoc::Left)),
            (Token::Min, OpKind::Binary) => Some(OpToken::new_binary(Token::Min, 1, Assoc::Left)),
            (Token::Min, OpKind::Prefix) => Some(OpToken::new_prefix(Token::Min, 1)),
            (Token::Mul, OpKind::Binary) => Some(OpToken::new_binary(Token::Mul, 2, Assoc::Left)),
            _ => None,
        }
    }

    fn combine(lhs: Option<Self>, op: OpToken<Self::Token>, rhs: Option<Self>) -> Self {
        match (lhs, op.token, rhs) {
            (Some(lhs), Token::Plus, Some(rhs)) => Self::Add(Box::new(lhs), Box::new(rhs)),
            (Some(lhs), Token::Min, Some(rhs)) => Self::Sub(Box::new(lhs), Box::new(rhs)),
            (None, Token::Min, Some(rhs)) => Self::Sub(Box::new(Self::Just(0.0)), Box::new(rhs)),
            (Some(lhs), Token::Mul, Some(rhs)) => Self::Mul(Box::new(lhs), Box::new(rhs)),
            _ => unreachable!(),
        }
    }

    fn operand_parser() -> impl Parser<Token = Token, Expression = Self> {
        just!(Token::Price(_f) => _f).map(|f| Self::Just(f))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Amount(pub f64);

impl Amount {
    pub fn get(&self) -> f64 {
        self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0.abs() < f64::EPSILON
    }

    pub fn parser() -> impl Parser<Token = Token, Expression = Amount> {
        just!(Token::DollarSymbol)
            .then(precedence::<ComplexAmount>())
            // .then(just!(Token::Price(_w) => _w))
            .map(|(_, a)| Amount(a.eval()))
    }
}

impl Eq for Amount {}
impl Ord for Amount {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Person(String);

impl Person {
    pub fn name(&self) -> &str {
        &self.0
    }

    pub fn parser() -> impl Parser<Token = Token, Expression = Person> {
        just!(Token::NameAnchor)
            .then(just!(Token::Word(_w) => _w))
            .map(|(_, name)| Person(name))
    }
}

#[derive(Debug)]
pub enum Debtor {
    EveryoneBut(HashSet<Person>),
    Only(HashSet<Person>),
}

impl Debtor {
    pub fn list_persons_given<'a>(&'a self, everyone: &'a HashSet<Person>) -> HashSet<Person> {
        match self {
            Self::EveryoneBut(blacklist) => everyone.difference(blacklist).cloned().collect(),
            Self::Only(whitelist) => whitelist.clone(),
        }
    }

    pub fn parser() -> impl Parser<Token = Token, Expression = Debtor> {
        one_of([
            sequence([Token::KeywordEveryone, Token::KeywordBut])
                .then(list_of(Token::Comma, Person::parser()))
                .map(|(_, v)| Debtor::EveryoneBut(v.into_iter().collect()))
                .boxed(),
            just!(Token::KeywordEveryone)
                .map(|_| Debtor::EveryoneBut(HashSet::new()))
                .boxed(),
            list_of(Token::Comma, Person::parser())
                .map(|v| Debtor::Only(v.into_iter().collect()))
                .boxed(),
        ])
    }
}

#[derive(Debug)]
pub struct LedgerSection(pub Vec<LedgerEntry>);

impl LedgerSection {
    fn parser() -> impl Parser<Token = Token, Expression = Self> {
        just!(Token::Word(_s) if _s == "transactions")
            .then(just!(Token::SectionMarker))
            .then(just!(Token::LineEnd))
            .then(list_of(Token::LineEnd, LedgerEntry::parser()))
            .map(|unwind!(entries, _, _, _)| Self(entries))
    }
}

#[derive(Debug)]
pub struct LedgerEntry(pub Person, pub Amount, pub Debtor);

impl LedgerEntry {
    fn parser() -> impl Parser<Token = Token, Expression = LedgerEntry> {
        just!(Token::Min)
            .then(Person::parser())
            .tail()
            .then(just!(Token::KeywordPaid))
            .pop()
            .then(Amount::parser())
            .then(just!(Token::KeywordFor))
            .pop()
            .then(Debtor::parser())
            .then(just!(Token::Comment))
            .pop()
            .map(|unwind!(debtor, amount, person)| LedgerEntry(person, amount, debtor))
    }
}

#[derive(Debug)]
pub struct PersonSection(pub Vec<Person>);

impl PersonSection {
    fn parser() -> impl Parser<Token = Token, Expression = Self> {
        just!(Token::Word(_s) if _s == "persons")
            .then(just!(Token::SectionMarker))
            .then(just!(Token::LineEnd))
            .then(list_of(
                Token::LineEnd,
                just!(Token::Min)
                    .then(Person::parser())
                    .map(|unwind!(p, _)| p),
            ))
            .map(|unwind!(persons, _, _, _)| PersonSection(persons))
    }
}

#[cfg(test)]
mod tests {

    use crate::parser::tokenizer;

    use super::*;

    #[test]
    fn test_token_eq() {
        assert_eq!(Token::DollarSymbol, Token::DollarSymbol);
        assert_eq!(Token::Word("a".to_owned()), Token::Word("a".to_owned()));
        assert_ne!(Token::Word("a".to_owned()), Token::Word("b".to_owned()));
    }

    #[test]
    fn test_comment() {
        let mut tokens = tokenizer().tokenize("(salut)");

        assert!(matches!(tokens.next().unwrap().token, Token::Comment));

        assert!(tokens.next().is_none());
    }

    #[test]
    fn test_person() {
        let mut tokens = tokenizer().tokenize("@Foo");
        let mut parser = Person::parser();

        assert!(matches!(
            parser.run_to_completion(&mut tokens),
            Ok(Person(n)) if n == "Foo"
        ));
    }

    #[test]
    fn test_ledger_entry() {
        let mut parser = LedgerEntry::parser();
        let mut tokens =
            tokenizer().tokenize("- @Foo paid $-5+5*5+5 for everyone but @Bar (no reason)\n");
        let res = parser.run_to_completion(&mut tokens);
        dbg!(&res);
        assert!(matches!(
            res,
            Ok(LedgerEntry(_, Amount(25.0), Debtor::EveryoneBut(_)))
        ));
    }

    #[test]
    fn test_debtor() {
        let mut parser = Debtor::parser();
        let alex = Person("Alex".to_string());
        let toto = Person("Toto".to_string());

        let mut tokens = tokenizer().tokenize("everyone but @Alex");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            res,
            Ok(Debtor::EveryoneBut(v)) if v == HashSet::from([alex.clone()])
        ));

        let mut tokens = tokenizer().tokenize("everyone but @Alex, @Toto");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            res,
            Ok(Debtor::EveryoneBut(v)) if v == HashSet::from([alex.clone(), toto.clone()])
        ));

        let mut tokens = tokenizer().tokenize("@Alex, @Toto");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            res,
            Ok(Debtor::Only(v)) if v == HashSet::from([alex, toto])
        ));

        let mut tokens = tokenizer().tokenize("everyone");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            dbg!(res),
            Ok(Debtor::EveryoneBut(v)) if v == HashSet::new()
        ));
    }
}
