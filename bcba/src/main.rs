use clap::Parser;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

mod parser;
use parser::{tokenizer, Amount, Expression, LedgerSection, LedgerEntry, LedgerParser, Person, PersonSection};

#[derive(Parser)]
#[command(version)]
struct CliArgs {
    file: Option<PathBuf>,
    #[arg(short, long)]
    pretty: bool,
}

fn main() -> Result<(), String> {
    let args = CliArgs::parse();
    let expressions = parse(args.file.unwrap_or(PathBuf::from("Ledgerfile")))?;
    let mut payments: Vec<Payment> = vec![];

    let mut ledger = expressions
        .into_iter()
        .fold(Ledger::default(), |mut ledger, ex| {
            match ex {
                Expression::PersonSection(PersonSection(persons)) => {
                    for p in persons {
                        ledger.add_person(p);
                    }
                }
                Expression::LedgerSection(LedgerSection(entries)) => {
                    for LedgerEntry(creditor, amount, debtor_list) in entries {
                        let all_debtors = debtor_list.list_persons_given(&ledger.everyone);
                        let div = all_debtors.len() as f64;
                        for debtor in all_debtors {
                            ledger.add_expense(creditor.clone(), amount.get() / div, debtor.clone());
                        }
                    }
                }
                _ => {}
            }
            ledger
        });

    loop {
        match (ledger.largest_creditor(), ledger.largest_debtor()) {
            (Some((creditor, a)), Some((debtor, b))) => {
                let diff = Amount(a.get().min(-b.get()));
                if diff.is_zero() {
                    break;
                }
                payments.push(Payment {
                    from: debtor.name().to_string(),
                    amount: diff.get(),
                    to: creditor.name().to_string(),
                });
                ledger.add_expense(debtor, diff.get(), creditor);
            }
            (_, _) => panic!(),
        }
    }

    println!(
        "{}",
        if args.pretty {
            serde_json::to_string_pretty(&payments).unwrap()
        } else {
            serde_json::to_string(&payments).unwrap()
        }
    );

    Ok(())
}

fn parse(file: impl AsRef<Path>) -> Result<Vec<Expression>, String> {
    let program = std::fs::read_to_string(file).expect("File not found");
    let mut tokens = tokenizer().tokenize(&program);
    LedgerParser::parse(&mut tokens)
}

#[derive(Serialize)]
struct Payment {
    from: String,
    amount: f64,
    to: String,
}

#[derive(Default)]
struct Ledger {
    balances: HashMap<Person, Amount>,
    everyone: HashSet<Person>,
}

impl Ledger {
    fn add_person(&mut self, person: Person) {
        self.balances.insert(person.clone(), Amount(0.0));
        self.everyone.insert(person);
    }

    fn add_expense(&mut self, creditor: Person, amount: f64, debtor: Person) {
        let Some(current) = self.balances.get_mut(&creditor) else { panic!("Unknown person {}", creditor.name()) };
        *current = Amount(current.get() + amount);
        let Some(current) = self.balances.get_mut(&debtor) else { panic!("Unknown person {}", debtor.name()) };
        *current = Amount(current.get() - amount);
    }

    fn largest_creditor(&self) -> Option<(Person, Amount)> {
        self.balances
            .iter()
            .max_by_key(|(_, &balance)| balance)
            .map(|(k, v)| (k.clone(), v.clone()))
    }

    fn largest_debtor(&self) -> Option<(Person, Amount)> {
        self.balances
            .iter()
            .min_by_key(|(_, &balance)| balance)
            .map(|(k, v)| (k.clone(), v.clone()))
    }
}
