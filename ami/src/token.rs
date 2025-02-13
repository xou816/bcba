use std::{fmt::Display, marker::PhantomData};

use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Annotated<Token> {
    pub token: Token,
    pub row: usize,
    pub col: usize,
}

impl <T: Display> Annotated<T> {
    pub fn describe(&self) -> String {
        format!("{} at ln {}, col {}", self.token, self.row, self.col)
    }
}

#[derive(Default)]
pub struct Buffer {
    buffer: String,
    pos: (usize, usize),
    until: Option<char>,
}

impl Buffer {
    pub fn done<T>(&mut self, f: impl FnOnce(String) -> T) -> Option<T> {
        self.until = None;
        Some(f(std::mem::take(&mut self.buffer)))
    }

    pub fn push(&mut self, s: &str) -> &mut Self {
        self.buffer.push_str(s);
        self
    }

    pub fn until<T>(&mut self, c: char) -> Option<T> {
        self.until = Some(c);
        None
    }

    pub fn until_done<T>(&mut self, c: char, f: impl FnOnce(String) -> T) -> Option<T> {
        match self.until {
            Some(_) => {
                self.until = None;
                Some(f(std::mem::take(&mut self.buffer)))
            }
            None => {
                self.until = Some(c);
                None
            }
        }
    }


    pub fn buffering(&self) -> bool {
        self.until.is_some()
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

                let is_whitespace = word != "\n" && word.chars().all(char::is_whitespace);
                if is_whitespace && !self.buffer.buffering() {
                    return None;
                }

                if !self.buffer.buffering() || self.buffer.until == word.chars().next() {
                    let token = P::tokenize(word, &mut self.buffer)?;
                    let (row, col) = self.buffer.pos;
                    Some(Annotated { token, row, col })
                } else {
                    self.buffer.push(word);
                    None
                }
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
            .tokenize("if true {\nprint(\"hellô  world\")\n}")
            .collect();
        let expected = vec![
            Token::If.at(1, 1),
            Token::True.at(1, 4),
            Token::BraceOpen.at(1, 9),
            Token::LineEnd.at(1, 10),
            Token::Identifier("print".to_owned()).at(2, 1),
            Token::ParenOpen.at(2, 6),
            Token::LitString("hellô  world".to_owned()).at(2, 7),
            Token::ParenClose.at(2, 21),
            Token::LineEnd.at(2, 22),
            Token::BraceClose.at(3, 1),
        ];
        assert_eq!(expected, tokens);
    }
}
