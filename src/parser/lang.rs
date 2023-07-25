use std::cmp::Ordering;
use std::collections::HashSet;

use super::parser::*;
use super::token::*;

pub struct LedgerParser;

impl LedgerParser {
    pub fn parse(
        tokens: &mut dyn Iterator<Item = AnnotatedToken>,
    ) -> Result<Vec<Expression>, String> {
        let pep = PersonExpressionParser::make()
            .suffixed(Token::LineEnd)
            .map(Expression::PersonDeclaration)
            .boxed();
        let lep = LedgerEntryParser::make()
            .map(Expression::LedgerEntry)
            .boxed();
        let nil = Parsers::always_completing(|| Expression::None).boxed();
        let mut parser = Parsers::one_of(vec![lep, pep, nil]);
        parser.run_to_exhaustion(tokens)
    }
}

#[derive(Debug)]
pub enum Expression {
    PersonDeclaration(Person),
    LedgerEntry(LedgerEntry),
    None
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
        SequenceParserBuilder::new()
            .expect(Token::DollarSymbol)
            .save(|t| matches!(t, Token::Word(_)))
            .combine::<_, Token, Amount>(|t| {
                let Some(Token::Word(t)) = t.into_iter().next() else { panic!() };
                match t.parse::<f32>() {
                    Ok(price) => Ok(Amount(price)),
                    Err(e) => Err(format!("Failed to parse f32: {}", e)),
                }
            })
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
        SequenceParserBuilder::new()
            .expect(Token::NameAnchor)
            .save(|t| matches!(t, Token::Word(_)))
            .combine::<_, Token, Person>(|tokens| {
                let Some(Token::Word(name)) = tokens.into_iter().next() else { panic!("Expected a name") };
                Ok(Person(name))
            })
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

struct DebtorParser(Option<Debtor>, BoxedParser<Vec<Person>>);

impl DebtorParser {
    fn new() -> Self {
        Self(
            None,
            ListParser::new(Token::Comma, PersonExpressionParser::make()).boxed(),
        )
    }

    fn parse_persons(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Debtor> {
        let list = match self.0.as_mut() {
            Some(Debtor::EveryoneBut(blacklist)) => blacklist,
            Some(Debtor::Only(whitelist)) => whitelist,
            _ => panic!(),
        };
        match self.1.consume(token, next_token) {
            ParseResult::Accepted => ParseResult::Accepted,
            ParseResult::Complete(persons) => {
                *list = persons.into_iter().collect();
                if next_token.is_none()
                    || matches!(next_token, Some(AnnotatedToken { token, .. }) if token != &Token::Comma)
                {
                    ParseResult::Complete(self.0.take().unwrap())
                } else {
                    ParseResult::Accepted
                }
            }
            ParseResult::Failed(e) => self.fail(e),
            ParseResult::Ignored(_) => panic!(),
        }
    }
}

impl Parser for DebtorParser {
    type Expression = Debtor;

    fn try_consume(
        &mut self,
        token: AnnotatedToken,
        next_token: Option<&AnnotatedToken>,
    ) -> ParseResult<Self::Expression> {
        let everyone = "everyone";
        let but = "but";
        match (self.0.as_mut(), token, next_token) {
            (
                None,
                AnnotatedToken {
                    token: Token::Word(w),
                    ..
                },
                Some(AnnotatedToken { token: next, .. }),
            ) if w == everyone && next != &Token::Word(but.to_string()) => {
                self.complete(Debtor::EveryoneBut(HashSet::new()))
            }
            (
                None,
                AnnotatedToken {
                    token: Token::Word(w),
                    ..
                },
                None,
            ) if w == everyone => self.complete(Debtor::EveryoneBut(HashSet::new())),
            (
                None,
                AnnotatedToken {
                    token: Token::Word(w),
                    ..
                },
                Some(AnnotatedToken { token: next, .. }),
            ) if w == everyone && next == &Token::Word(but.to_string()) => ParseResult::Accepted,
            (
                None,
                AnnotatedToken {
                    token: Token::Word(w),
                    ..
                },
                _,
            ) if w == but => {
                self.0.replace(Debtor::EveryoneBut(HashSet::new()));
                ParseResult::Accepted
            }
            (
                None,
                t @ AnnotatedToken {
                    token: Token::NameAnchor,
                    ..
                },
                next_token,
            ) => {
                self.0.replace(Debtor::Only(HashSet::new()));
                self.parse_persons(t, next_token)
            }
            (Some(Debtor::EveryoneBut(_) | Debtor::Only(_)), token, next_token) => {
                self.parse_persons(token, next_token)
            }
            (_, _, _) => ParseResult::Failed("Unexpected input".to_string()),
        }
    }

    fn reset(&mut self) {
        self.1.reset();
    }
}

#[derive(Debug)]
pub struct LedgerEntry(pub Person, pub Amount, pub Debtor);

struct LedgerEntryParser;
enum LedgerToken {
    Creditor(Person),
    Debtor(Debtor),
    Amount(Amount),
}

impl TryFrom<Token> for LedgerToken {
    type Error = ();

    fn try_from(_: Token) -> Result<Self, Self::Error> {
        Err(())
    }
}

impl LedgerEntryParser {
    fn make() -> impl Parser<Expression = LedgerEntry> {
        SequenceParserBuilder::new()
            .expect(Token::LedgerEntryStart)
            .delegate(PersonExpressionParser::make().map(LedgerToken::Creditor))
            .expect(Token::KeywordPaid)
            .delegate(AmountExpressionParser::make().map(LedgerToken::Amount))
            .expect(Token::KeywordFor)
            .delegate(DebtorParser::new().map(LedgerToken::Debtor))
            .expect(Token::CommentStart)
            .save_until(|t| matches!(t, Token::CommentEnd))
            .expect(Token::LineEnd)
            .combine::<_, LedgerToken, LedgerEntry>(|tokens| {
                let mut t = tokens.into_iter();
                let Some(LedgerToken::Creditor(c)) = t.next() else { panic!("Expected a creditor") };
                let Some(LedgerToken::Amount(a)) = t.next() else { panic!("Expected an amount") };
                let Some(LedgerToken::Debtor(d)) = t.next() else { panic!("Expected a debtor") };
                Ok(LedgerEntry(c, a, d))
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
        let mut parser = DebtorParser::new();
        let alex = Person("Alex".to_string());
        let toto = Person("Toto".to_string());

        let mut tokens = Tokenizer::tokenize("everyone but @Alex");
        let res = parser.run_to_completion(&mut tokens);

        assert!(matches!(
            res,
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
