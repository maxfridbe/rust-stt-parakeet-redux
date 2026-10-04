use serde::Deserialize;
use std::collections::HashMap;

use crate::{Result, error::invalid};

#[derive(Deserialize)]
struct TokenizerFile {
    model: Vocabulary,
    decoder: serde_json::Value,
    added_tokens: Vec<AddedToken>,
}
#[derive(Deserialize)]
struct Vocabulary {
    vocab: HashMap<String, usize>,
}
#[derive(Deserialize)]
struct AddedToken {
    id: usize,
    content: String,
    special: bool,
}

pub(crate) struct Tokenizer {
    pieces: Vec<String>,
    special: Vec<bool>,
}

impl Tokenizer {
    pub fn load(bytes: &[u8], vocabulary_size: usize, blank: usize) -> Result<Self> {
        let file: TokenizerFile = serde_json::from_slice(bytes)?;
        if file.decoder["type"] != "Metaspace"
            || file.decoder["replacement"] != "▁"
            || file.decoder["prepend_scheme"] != "always"
        {
            return Err(invalid("expected the Parakeet Metaspace tokenizer"));
        }
        let mut pieces = vec![String::new(); vocabulary_size];
        let mut special = vec![false; vocabulary_size];
        for (piece, id) in file.model.vocab {
            let target = pieces
                .get_mut(id)
                .ok_or_else(|| invalid("tokenizer ID exceeds vocabulary size"))?;
            *target = piece;
        }
        for added in file.added_tokens {
            let target = pieces
                .get_mut(added.id)
                .ok_or_else(|| invalid("added token exceeds vocabulary size"))?;
            *target = added.content;
            special[added.id] = added.special;
        }
        if pieces.iter().any(String::is_empty) || pieces[blank] != "<blank>" {
            return Err(invalid("incomplete vocabulary or mismatched blank ID"));
        }
        special[blank] = true;
        Ok(Self { pieces, special })
    }

    pub fn piece(&self, id: usize) -> Option<&str> {
        (!self.special[id]).then_some(self.pieces[id].as_str())
    }

    pub fn decode(&self, ids: impl IntoIterator<Item = usize>) -> String {
        let mut text = String::new();
        for id in ids {
            if let Some(piece) = self.piece(id) {
                text.push_str(&piece.replace('▁', " "));
            }
        }
        text.strip_prefix(' ').unwrap_or(&text).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_subwords_and_preserves_unicode_and_punctuation() {
        let tokenizer = Tokenizer {
            pieces: vec![
                "▁Hello".into(),
                ",".into(),
                "▁世".into(),
                "界".into(),
                "<blank>".into(),
            ],
            special: vec![false, false, false, false, true],
        };
        assert_eq!(tokenizer.decode([0, 1, 2, 3, 4]), "Hello, 世界");
    }
}
