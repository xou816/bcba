use std::iter::once;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct AnnotatedToken {
    pub token: Token,
    pub row: usize,
    pub col: usize,
}

#[derive(Debug, Eq, Clone)]
pub enum Token {
    NameAnchor,
    LedgerEntryStart,
    DollarSymbol,
    CommentStart,
    CommentEnd,
    KeywordPaid,
    KeywordFor,
    LineEnd,
    Comma,
    Word(String),
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Word(l), Self::Word(r)) => l == r,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Token {
    pub fn display_name(&self) -> String {
        match self {
            Token::LedgerEntryStart => "start of ledger".to_string(),
            Token::KeywordPaid => "keyword `paid`".to_string(),
            Token::NameAnchor => "name anchor `@`".to_string(),
            Token::Word(w) => format!("token `{}`", w),
            Token::DollarSymbol => "currency tag `$`".to_string(),
            Token::CommentStart => "start of comment `(`".to_string(),
            Token::CommentEnd => "start of comment `)`".to_string(),
            Token::KeywordFor => "keyword `for`".to_string(),
            Token::LineEnd => "end of line".to_string(),
            Token::Comma => "comma".to_string(),
        }
    }

    fn try_from_char(c: char) -> Option<Token> {
        match c {
            '@' => Some(Token::NameAnchor),
            '-' => Some(Token::LedgerEntryStart),
            '$' => Some(Token::DollarSymbol),
            '(' => Some(Token::CommentStart),
            ')' => Some(Token::CommentEnd),
            '\n' => Some(Token::LineEnd),
            ',' => Some(Token::Comma),
            _ => None,
        }
    }

    fn for_word(str: &str) -> Token {
        match str {
            "paid" => Token::KeywordPaid,
            "for" => Token::KeywordFor,
            _ => Token::Word(str.to_string()),
        }
    }

    pub fn at(self, row: usize, col: usize) -> AnnotatedToken {
        AnnotatedToken {
            token: self,
            row,
            col,
        }
    }
}

pub struct Tokenizer {
    buffer: [u8; 32],
    buffer_pos: usize,
    row: usize,
    col_offset: usize,
}

impl Tokenizer {
    fn new() -> Self {
        Self {
            buffer: [0u8; 32],
            buffer_pos: 0,
            row: 1,
            col_offset: 0,
        }
    }

    fn row_col(&self, i: usize) -> (usize, usize) {
        let col = i - self.buffer_pos - self.col_offset + 1;
        (self.row, col)
    }

    fn bump_row(&mut self, i: usize) {
        self.row += 1;
        self.col_offset = i + 1;
    }

    fn consume_buffer(&mut self, i: usize) -> Option<AnnotatedToken> {
        if self.buffer_pos == 0 {
            return None;
        }

        let (row, col) = self.row_col(i);
        let content = std::mem::replace(&mut self.buffer, [0; 32]);
        let content = std::str::from_utf8(&content[..self.buffer_pos]).expect("Encoding error");
        self.buffer_pos = 0;
        Some(Token::for_word(content).at(row, col))
    }

    fn accept(&mut self, i: usize, chr: char) -> Vec<AnnotatedToken> {
        let token = Token::try_from_char(chr);
        if chr == ' ' || token.is_some() {
            let buffer_token = self.consume_buffer(i);
            let token = token.map(|t| {
                let (row, col) = self.row_col(i);
                if let Token::LineEnd = t {
                    self.bump_row(i);
                }
                t.at(row, col)
            });
            match (buffer_token, token) {
                (Some(a), Some(b)) => vec![a, b],
                (None, Some(a)) | (Some(a), None) => vec![a],
                _ => vec![],
            }
        } else {
            self.buffer[self.buffer_pos] = chr.try_into().expect("Encoding error");
            self.buffer_pos = self.buffer_pos + 1;
            vec![]
        }
    }

    pub fn tokenize(program: &str) -> impl Iterator<Item = AnnotatedToken> + '_ {
        program
            .char_indices()
            .chain(once((program.len(), ' ')))
            .scan(Self::new(), |tokenizer, (i, chr)| {
                Some(tokenizer.accept(i, chr))
            })
            .flat_map(|it| it.into_iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_eq() {
        assert_eq!(Token::DollarSymbol, Token::DollarSymbol);
        assert_ne!(Token::CommentStart, Token::CommentEnd);
        assert_eq!(Token::Word("a".to_owned()), Token::Word("a".to_owned()));
        assert_ne!(Token::Word("a".to_owned()), Token::Word("b".to_owned()));
    }

    #[test]
    fn test_tokenizer() {
        let tokens: Vec<Token> = Tokenizer::tokenize("@Foo paid $4 for something")
            .map(|t| t.token)
            .collect();
        let expected = vec![
            Token::NameAnchor,
            Token::Word("Foo".to_owned()),
            Token::KeywordPaid,
            Token::DollarSymbol,
            Token::Word("4".to_owned()),
            Token::KeywordFor,
            Token::Word("something".to_owned()),
        ];
        assert_eq!(expected, tokens);
    }

    #[test]
    fn test_annotated() {
        let tokens: Vec<AnnotatedToken> = Tokenizer::tokenize("@Foo\nbar").collect();
        let expected = vec![
            Token::NameAnchor.at(1, 1),
            Token::Word("Foo".to_owned()).at(1, 2),
            Token::LineEnd.at(1, 5),
            Token::Word("bar".to_owned()).at(2, 1),
        ];
        assert_eq!(expected, tokens);
    }
}
