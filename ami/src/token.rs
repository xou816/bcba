use std::iter::once;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Annotated<Token> {
    pub token: Token,
    pub row: usize,
    pub col: usize,
}

pub trait TokenDeserialize where Self: Sized {
    fn try_from_char(c: char) -> Option<Self>;
    fn for_word(str: &str) -> Self;    

    fn at(self, row: usize, col: usize) -> Annotated<Self> {
        Annotated {
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

    fn consume_buffer<T: TokenDeserialize>(&mut self, i: usize) -> Option<Annotated<T>> {
        if self.buffer_pos == 0 {
            return None;
        }

        let (row, col) = self.row_col(i);
        let content = std::mem::replace(&mut self.buffer, [0; 32]);
        let content = std::str::from_utf8(&content[..self.buffer_pos]).expect("Encoding error");
        self.buffer_pos = 0;
        Some(T::for_word(content).at(row, col))
    }

    fn accept<T: TokenDeserialize>(&mut self, i: usize, chr: char) -> Vec<Annotated<T>> {
        let token = T::try_from_char(chr);
        if chr == ' ' || token.is_some() {
            let buffer_token = self.consume_buffer(i);
            let token = token.map(|token| {
                let (row, col) = self.row_col(i);
                if chr == '\n' {
                    self.bump_row(i);
                }
                token.at(row, col)
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

    pub fn tokenize<T: TokenDeserialize + 'static>(program: &str) -> impl Iterator<Item = Annotated<T>> + '_ {
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

    enum Token {
        
    }

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
        let tokens: Vec<Annotated<Token>> = Tokenizer::tokenize("@Foo\nbar").collect();
        let expected = vec![
            Token::NameAnchor.at(1, 1),
            Token::Word("Foo".to_owned()).at(1, 2),
            Token::LineEnd.at(1, 5),
            Token::Word("bar".to_owned()).at(2, 1),
        ];
        assert_eq!(expected, tokens);
    }
}
