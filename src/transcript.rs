use serde::Serialize;

use crate::{decoder::Emission, tokenizer::Tokenizer};

/// A decoded subword and its TDT alignment in seconds.
#[derive(Debug, Clone, Serialize)]
pub struct Token {
    pub id: usize,
    pub text: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// A sentence, or the remaining words at the end of an audio chunk.
#[derive(Debug, Clone, Serialize)]
pub struct Segment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub words: Vec<Word>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Transcript {
    pub text: String,
    pub duration_seconds: f64,
    pub segments: Vec<Segment>,
    pub tokens: Vec<Token>,
}

impl Transcript {
    pub(crate) fn append(
        &mut self,
        emissions: &[Emission],
        tokenizer: &Tokenizer,
        offset: f64,
        end: f64,
    ) {
        let text = tokenizer.decode(emissions.iter().map(|emission| emission.token_id));
        if !self.text.is_empty() && !text.is_empty() {
            self.text.push(' ');
        }
        self.text.push_str(&text);
        let mut words: Vec<Word> = Vec::new();
        for emission in emissions {
            let Some(piece) = tokenizer.piece(emission.token_id) else {
                continue;
            };
            let start = (offset + emission.frame as f64 * 0.08).min(end);
            let stop =
                (offset + emission.frame.saturating_add(emission.duration) as f64 * 0.08).min(end);
            let text = piece.replace('▁', " ");
            self.tokens.push(Token {
                id: emission.token_id,
                text: text.clone(),
                start,
                end: stop,
            });
            append_word(&mut words, &text, start, stop);
        }
        self.segments.extend(sentences(words));
    }
}

fn append_word(words: &mut Vec<Word>, piece: &str, start: f64, end: f64) {
    let stripped = piece.trim();
    if stripped.is_empty() {
        return;
    }
    let punctuation = stripped
        .chars()
        .all(|character| ".,!?;:…。，！？、：；'’\"”)]}»".contains(character));
    if let Some(word) = words
        .last_mut()
        .filter(|_| !piece.starts_with(' ') || punctuation)
    {
        word.word.push_str(stripped);
        word.end = end;
        return;
    }
    words.push(Word {
        word: stripped.into(),
        start,
        end,
    });
}

fn sentences(words: Vec<Word>) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut pending = Vec::new();
    for word in words {
        let ends_sentence = word.word.ends_with(['.', '?', '!', '。', '？', '！']);
        pending.push(word);
        if ends_sentence {
            flush_sentence(&mut segments, &mut pending);
        }
    }
    flush_sentence(&mut segments, &mut pending);
    segments
}

fn flush_sentence(segments: &mut Vec<Segment>, words: &mut Vec<Word>) {
    let (Some(first), Some(last)) = (words.first(), words.last()) else {
        return;
    };
    segments.push(Segment {
        text: words
            .iter()
            .map(|word| word.word.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        start: first.start,
        end: last.end,
        words: std::mem::take(words),
    });
}
