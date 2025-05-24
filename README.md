# bcba

Balance expenses between people using a plain text file. Version that file, maybe. _Les Bons Comptes font les Bons Amis._

## Sample Ledgerfile

Start by listing involved people in a file named `Ledgerfile`.

```
persons:
- @Foo
- @Bar
- @Baz
```

Then list expenses.

```
transactions:
- @Foo paid $50 for everyone (because he felt generous)
- @Bar paid $20 for everyone but @Foo (because he hates Foo's guts)
```

Run the program. It will list the required payments to balance expenses among Foo, Bar and Baz.

```json
[{"from":"Baz","amount":26.666666,"to":"Foo"},{"from":"Bar","amount":6.666666,"to":"Foo"}]
```

Profit.

# ami, the friendly parser

This project is just an excuse to build a basic parser library. See the included example or `bcba` for usage samples. 

Here is for instance the parser for a single ledger entry in `bcba`:

```rust
just!(Token::ListItem)
    .then(Person::parser())
    .then(just!(Token::KeywordPaid))
    .then(Amount::parser())
    .then(just!(Token::KeywordFor))
    .then(Debtor::parser())
    .then(just!(Token::Comment))
    .map(|unwind!(_, debtor, _, amount, _, person, _)| LedgerEntry(person, amount, debtor))
```

Breakdown of interesting things:
- (not pictured here) a plain enum defines all our tokens
- the `just!` macro turns a token in a parser that parses that specific token
- parsers are combined with operators just as `then`, `map`, or other **combinators** (these are WIP and subject to change, a lot)
- in this example, other parsers are referenced (e.g. `Person::parser()`)
- map is a bit wonky, therefore an `unwind!` macro is used to unwrap (in reverse order!) the elements parsed by each of the chained parsers: first, a comment (ignored), then a debtor, etc... 