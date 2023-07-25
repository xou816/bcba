# bcba

Balance expenses between people using a plain text file. Version that file, maybe. _Les Bons Comptes font les Bons Amis._

## Sample Ledgerfile

Start by listing involved people in a file named `Ledgerfile`.

```
@Foo
@Bar
@Baz
```

Then list expenses.

```
- @Foo paid $50 for everyone (because he felt generous)
- @Bar paid $20 for everyone but @Foo (because he hates Foo's guts)
```

Run the program. It will list the required payments to balance expenses among Foo, Bar and Baz.

```json
[{"from":"Baz","amount":26.666666,"to":"Foo"},{"from":"Bar","amount":6.666666,"to":"Foo"}]
```

Profit.

## Parser

This project is just an excuse to build a basic parser library, check out `parser.rs`.

As for the actual ledger parser, it's got some quirks at the moment:
- files must end with a new line (`\n`);
- it does not handle UTF-8 graphemes properly or at all, stick to ASCII for now;
- `(comments like this)` are not optional;
- it `panics!` quite liberally (but is working on anxiety management).

## Future

It's just for fun. But multiple currencies would be nice, I guess.