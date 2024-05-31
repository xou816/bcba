use std::{iter::once, marker::PhantomData};

use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Annotated<Token> {
    pub token: Token,
    pub row: usize,
    pub col: usize,
}

pub trait TokenDeserialize
where
    Self: Sized,
{
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

#[deprecated]
pub struct _Tokenizer {
    buffer: [u8; 32],
    buffer_pos: usize,
    row: usize,
    col_offset: usize,
}

impl _Tokenizer {
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

    pub fn tokenize<T: TokenDeserialize + 'static>(
        program: &str,
    ) -> impl Iterator<Item = Annotated<T>> + '_ {
        program
            .char_indices()
            .chain(once((program.len(), ' ')))
            .scan(Self::new(), |tokenizer, (i, chr)| {
                Some(tokenizer.accept(i, chr))
            })
            .flat_map(|it| it.into_iter())
    }
}

#[derive(Default)]
pub struct Buffer {
    buffer: Vec<String>,
    pos: (usize, usize),
    buffering: bool,
}

impl Buffer {
    pub fn take<T>(&mut self, f: impl FnOnce(Vec<String>) -> T) -> Option<T> {
        self.buffering = false;
        Some(f(self.buffer.drain(..).collect()))
    }

    pub fn push(&mut self, s: &str) -> &mut Self {
        self.buffering = true;
        self.buffer.push(s.to_string());
        self
    }

    pub fn expect<T>(&mut self) -> Option<T> {
        self.buffering = true;
        None
    }

    pub fn buffering(&self) -> bool {
        self.buffering
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

pub enum Tokenize<T> {
    Buffer,
    Yield(T),
}

pub trait TokenProducer {
    type Token;
    fn tokenize(word: &str, buffer: &mut Buffer) -> Option<Self::Token>;
}

pub struct Tokenizer<P> {
    producer: PhantomData<P>,
    buffer: Buffer,
    col: usize,
    row: usize,
}

impl<P> Tokenizer<P>
where
    P: TokenProducer + 'static,
{
    pub fn new() -> Self {
        Self {
            producer: PhantomData,
            buffer: Default::default(),
            col: 1,
            row: 1,
        }
    }

    fn compute_pos(&mut self, word: &str) {
        if !self.buffer.buffering() {
            self.buffer.pos = (self.row, self.col);
        }
        match word {
            "\n" => {
                self.col = 1;
            self.row += 1;
            },
            _ => self.col += word.graphemes(true).count(),
        }
    }

    pub fn tokenize(mut self, program: &str) -> impl Iterator<Item = Annotated<P::Token>> + '_ {
        program
            .split_word_bounds()
            .into_iter()
            .filter_map(move |word| {
                self.compute_pos(word);

                if word != "\n" && word.chars().all(char::is_whitespace) {
                    return None;
                }
                let token = P::tokenize(word, &mut self.buffer)?;
                let (row, col) = self.buffer.pos;
                Some(Annotated { token, row, col })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toy::Token;

    #[test]
    fn test_v2() {
        let tokens: Vec<Annotated<Token>> = Tokenizer::<Token>::new()
            .tokenize("if true {\nprint(\"hellô world\")\n}")
            .collect();
        let expected = vec![
            Token::If.at(1, 1),
            Token::True.at(1, 4),
            Token::BraceOpen.at(1, 9),
            Token::LineEnd.at(1, 10),
            Token::Identifier("print".to_owned()).at(2, 1),
            Token::ParenOpen.at(2, 6),
            Token::LitString("hellô world".to_owned()).at(2, 7),
            Token::ParenClose.at(2, 20),
            Token::LineEnd.at(2, 21),
            Token::BraceClose.at(3, 1),
        ];
        assert_eq!(expected, tokens);
    }

    #[test]
    fn test_tokenizer() {
        let tokens: Vec<Token> = _Tokenizer::tokenize("if true { exit }")
            .map(|t| t.token)
            .collect();
        let expected = vec![
            Token::If,
            Token::True,
            Token::BraceOpen,
            Token::Identifier("exit".to_string()),
            Token::BraceClose,
        ];
        assert_eq!(expected, tokens);
    }

    #[test]
    fn test_annotated() {
        let tokens: Vec<Annotated<Token>> = _Tokenizer::tokenize("if true {\nexit\n}").collect();
        let expected = vec![
            Token::If.at(1, 1),
            Token::True.at(1, 4),
            Token::BraceOpen.at(1, 9),
            Token::LineEnd.at(1, 10),
            Token::Identifier("exit".to_owned()).at(2, 1),
            Token::LineEnd.at(2, 5),
            Token::BraceClose.at(3, 1),
        ];
        assert_eq!(expected, tokens);
    }
}
