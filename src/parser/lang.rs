use std::cmp::Ordering;
use std::collections::HashSet;

use self::parsers::{always_completing, expect, one_of};
use self::parsers::{expect_delimited, list_of};

use super::parser::*;
use super::token::*;

pub struct LedgerParser;

impl LedgerParser {
    pub fn parse(
        tokens: &mut dyn Iterator<Item = AnnotatedToken>,
    ) -> Result<Vec<Expression>, String> {
        let pep = PersonExpressionParser::make()
            .then(expect_token!(Token::LineEnd))
            .map(|(p, _)| Expression::PersonDeclaration(p))
            .boxed();
        let lep = LedgerEntryParser::make()
            .map(Expression::LedgerEntry)
            .boxed();
        let nil = always_completing(|| Expression::None).boxed();
        let mut parser = one_of(vec![lep, pep, nil]);
        parser.run_to_exhaustion(tokens)
    }
}

#[derive(Debug)]
pub enum Expression {
    PersonDeclaration(Person),
    LedgerEntry(LedgerEntry),
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Amount(pub f32);

impl Amount {
    pub fn get(&self) -> f32 {
        self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0.abs() < 1.0 * f32::EPSILON
    }
}

impl Eq for Amount {}
impl Ord for Amount {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

struct AmountExpressionParser;

impl AmountExpressionParser {
    fn make() -> impl Parser<Expression = Amount> {
        expect_token!(Token::DollarSymbol)
            .then(expect_token!(Token::Word(w) => w).try_map(|w| {
                match w.parse::<f32>() {
                    Ok(price) => Ok(Amount(price)),
                    Err(e) => Err(format!("Failed to parse f32: {}", e)),
                }
            }))
            .map(|(_, a)| a)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Person(String);

impl Person {
    pub fn name(&self) -> &str {
        &self.0
    }
}

struct PersonExpressionParser;

impl PersonExpressionParser {
    fn make() -> impl Parser<Expression = Person> {
        expect_token!(Token::NameAnchor)
            .then(expect_token!(Token::Word(w) => w))
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
}

struct DebtorParser;

impl DebtorParser {
    fn make() -> impl Parser<Expression = Debtor> {
        one_of(vec![
            expect([
                Token::Word("everyone".to_string()),
                Token::Word("but".to_string()),
            ])
                .then(list_of(Token::Comma, PersonExpressionParser::make()))
                .map(|(_, v)| Debtor::EveryoneBut(v.into_iter().collect()))
                .boxed(),
            list_of(Token::Comma, PersonExpressionParser::make())
                .map(|v| Debtor::Only(v.into_iter().collect()))
                .boxed(),
            expect_token!(Token::Word(ref w) if w == "everyone")
                .map(|_| Debtor::EveryoneBut(HashSet::new()))
                .boxed(),
        ])
    }
}

#[derive(Debug)]
pub struct LedgerEntry(pub Person, pub Amount, pub Debtor);

struct LedgerEntryParser;

impl LedgerEntryParser {
    fn make() -> impl Parser<Expression = LedgerEntry> {
        expect_token!(Token::LedgerEntryStart)
            .then(PersonExpressionParser::make())
            .then(expect_token!(Token::KeywordPaid))
            .then(AmountExpressionParser::make())
            .then(expect_token!(Token::KeywordFor))
            .then(DebtorParser::make())
            .then(expect_delimited([Token::CommentStart, Token::CommentEnd]))
            .then(expect_token!(Token::LineEnd))
            .map(|unwind!(_, _, debtor, _, amount, _, person, _)| {
                LedgerEntry(person, amount, debtor)
            })
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_person() {
        let mut tokens = Tokenizer::tokenize("@Foo");
        let mut parser = PersonExpressionParser::make();

        assert!(matches!(
            parser.run_to_completion(&mut tokens),
            Ok(Person(n)) if n == "Foo"
        ));
    }

    #[test]
    fn test_debtor() {
        let mut parser = DebtorParser::make();
        let alex = Person("Alex".to_string());
        let toto = Person("Toto".to_string());

        let mut tokens = Tokenizer::tokenize("everyone but @Alex");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            dbg!(res),
            Ok(Debtor::EveryoneBut(v)) if v == HashSet::from([alex.clone()])
        ));

        let mut tokens = Tokenizer::tokenize("everyone but @Alex, @Toto");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            res,
            Ok(Debtor::EveryoneBut(v)) if v == HashSet::from([alex.clone(), toto.clone()])
        ));

        let mut tokens = Tokenizer::tokenize("@Alex, @Toto");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            res,
            Ok(Debtor::Only(v)) if v == HashSet::from([alex, toto])
        ));

        let mut tokens = Tokenizer::tokenize("everyone");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            res,
            Ok(Debtor::EveryoneBut(v)) if v == HashSet::new()
        ));
    }
}
