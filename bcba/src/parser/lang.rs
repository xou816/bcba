use std::cmp::Ordering;
use std::collections::HashSet;

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
            LedgerEntry::parser().map(Expression::LedgerEntry).boxed(),
            Person::parser()
                .then(just!(Token::LineEnd))
                .map(|(p, _)| Expression::PersonDeclaration(p))
                .boxed(),
            just!(Token::LineEnd).map(|_| Expression::None).boxed(),
        ]);
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

    pub fn parser() -> impl Parser<Token = Token, Expression = Amount> {
        just!(Token::DollarSymbol)
            .then(
                just!(Token::Word(w) => w).try_map(|w| match w.parse::<f32>() {
                    Ok(price) => Ok(Amount(price)),
                    Err(e) => Err(format!("Failed to parse f32: {}", e)),
                }),
            )
            .map(|(_, a)| a)
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
            .then(just!(Token::Word(w) => w))
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
pub struct LedgerEntry(pub Person, pub Amount, pub Debtor);

impl LedgerEntry {
    fn parser() -> impl Parser<Token = Token, Expression = LedgerEntry> {
        just!(Token::LedgerEntryStart)
            .then(Person::parser())
            .then(just!(Token::KeywordPaid))
            .then(Amount::parser())
            .then(just!(Token::KeywordFor))
            .then(Debtor::parser())
            .then(just!(Token::Comment))
            .then(just!(Token::LineEnd))
            .map(|unwind!(_, _, debtor, _, amount, _, person, _)| {
                LedgerEntry(person, amount, debtor)
            })
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ami::token::Tokenizer;

    #[test]
    fn test_token_eq() {
        assert_eq!(Token::DollarSymbol, Token::DollarSymbol);
        assert_eq!(Token::Word("a".to_owned()), Token::Word("a".to_owned()));
        assert_ne!(Token::Word("a".to_owned()), Token::Word("b".to_owned()));
    }

    #[test]
    fn test_comment() {
        let mut tokens = Tokenizer::<Token>::new().tokenize("(salut)");

        assert!(matches!(tokens.next().unwrap().token, Token::Comment));

        assert!(tokens.next().is_none());
    }

    #[test]
    fn test_person() {
        let mut tokens = Tokenizer::<Token>::new().tokenize("@Foo");
        let mut parser = Person::parser();

        assert!(matches!(
            parser.run_to_completion(&mut tokens),
            Ok(Person(n)) if n == "Foo"
        ));
    }

    #[test]
    fn test_ledger_entry() {
        let mut parser = LedgerEntry::parser();
        let mut tokens = Tokenizer::<Token>::new()
            .tokenize("- @Foo paid $3.99 for everyone but @Bar (no reason)\n");
        let res = parser.run_to_completion(&mut tokens);
        assert!(matches!(res, Ok(LedgerEntry(_, _, Debtor::EveryoneBut(_)))));
    }

    #[test]
    fn test_debtor() {
        let mut parser = Debtor::parser();
        let alex = Person("Alex".to_string());
        let toto = Person("Toto".to_string());

        let mut tokens = Tokenizer::<Token>::new().tokenize("everyone but @Alex");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            res,
            Ok(Debtor::EveryoneBut(v)) if v == HashSet::from([alex.clone()])
        ));

        let mut tokens = Tokenizer::<Token>::new().tokenize("everyone but @Alex, @Toto");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            res,
            Ok(Debtor::EveryoneBut(v)) if v == HashSet::from([alex.clone(), toto.clone()])
        ));

        let mut tokens = Tokenizer::<Token>::new().tokenize("@Alex, @Toto");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            res,
            Ok(Debtor::Only(v)) if v == HashSet::from([alex, toto])
        ));

        let mut tokens = Tokenizer::<Token>::new().tokenize("everyone");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            dbg!(res),
            Ok(Debtor::EveryoneBut(v)) if v == HashSet::new()
        ));
    }
}
