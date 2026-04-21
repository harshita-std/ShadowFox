"""
Task 3B: Autocorrect Keyboard — Next-Word Prediction
======================================================
Predicts the next word using:
  (A) N-gram Language Model  (NLTK, fast, no GPU needed)
  (B) LSTM Neural Network    (TensorFlow/Keras, more powerful)

Requirements:
    pip install nltk tensorflow numpy
    python -c "import nltk; nltk.download('gutenberg'); nltk.download('punkt')"
"""

import re
import random
import pickle
from collections import defaultdict, Counter

import numpy as np
import nltk

# Download NLTK corpora if not already present
for resource in ("gutenberg", "punkt", "punkt_tab"):
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PART A — N-GRAM LANGUAGE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class NGramPredictor:
    """
    Language model based on N-grams (bigram / trigram / n-gram).

    Trains on any iterable of sentences (lists of tokens).
    Predicts top-K next words given a context of (n-1) preceding words.
    """

    def __init__(self, n=3):
        """
        Args:
            n : N-gram order (2 = bigram, 3 = trigram, …)
        """
        assert n >= 2, "N must be ≥ 2"
        self.n        = n
        self.ngrams   = defaultdict(Counter)   # context → {word: count}
        self.vocab    = set()
        self.trained  = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, sentences):
        """
        Args:
            sentences : Iterable of token lists, e.g. [["the", "cat", "sat"], …]
        """
        print(f"[INFO] Training {self.n}-gram model...")
        total = 0
        for sent in sentences:
            padded = ["<s>"] * (self.n - 1) + list(sent) + ["</s>"]
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i : i + self.n - 1])
                word    = padded[i + self.n - 1]
                self.ngrams[context][word] += 1
                self.vocab.add(word)
            total += 1

        self.trained = True
        print(f"  Sentences trained : {total}")
        print(f"  Unique contexts   : {len(self.ngrams)}")
        print(f"  Vocabulary size   : {len(self.vocab)}")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_next(self, context_words, top_k=5):
        """
        Predict the top-K next words.

        Args:
            context_words : List of preceding words (last n-1 used).
            top_k         : Number of suggestions to return.

        Returns:
            list of (word, probability) tuples, sorted by probability desc.
        """
        if not self.trained:
            raise RuntimeError("Model is not trained yet. Call .train() first.")

        # Use last (n-1) words as context; pad if shorter
        ctx  = ["<s>"] * (self.n - 1) + [w.lower() for w in context_words]
        ctx  = tuple(ctx[-(self.n - 1):])

        candidates = self.ngrams.get(ctx, None)

        # Backoff to lower-order n-gram if context unseen
        if not candidates and len(ctx) > 1:
            for order in range(len(ctx) - 1, 0, -1):
                shorter = ctx[-order:]
                candidates = self.ngrams.get(shorter, None)
                if candidates:
                    break

        if not candidates:
            return []

        total = sum(candidates.values())
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [(w, cnt / total) for w, cnt in ranked[:top_k]]

    def complete_sentence(self, seed_words, max_words=15, top_k=3):
        """
        Generate a sentence continuation by greedily sampling from top-K predictions.

        Args:
            seed_words : Starting context (list of words).
            max_words  : Maximum words to generate.
            top_k      : Pool size for random sampling.

        Returns:
            Full sentence string.
        """
        result = list(seed_words)
        for _ in range(max_words):
            preds = self.predict_next(result, top_k=top_k)
            if not preds:
                break
            # Weighted random choice from top predictions
            words  = [p[0] for p in preds]
            probs  = np.array([p[1] for p in preds])
            probs /= probs.sum()
            choice = np.random.choice(words, p=probs)
            if choice in ("</s>", "<s>"):
                break
            result.append(choice)
        return " ".join(result)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path="ngram_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"n": self.n, "ngrams": dict(self.ngrams), "vocab": self.vocab}, f)
        print(f"[INFO] N-gram model saved → {path}")

    def load(self, path="ngram_model.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n      = data["n"]
        self.ngrams = defaultdict(Counter, data["ngrams"])
        self.vocab  = data["vocab"]
        self.trained = True
        print(f"[INFO] N-gram model loaded ← {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — LSTM NEURAL NETWORK NEXT-WORD PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class LSTMNextWordPredictor:
    """
    LSTM-based next-word predictor built with TensorFlow/Keras.

    Architecture:
      Embedding → LSTM (128) → Dropout → Dense (softmax)

    Trained on fixed-length windows of tokens from any text corpus.
    """

    def __init__(self, seq_len=10, embed_dim=64, lstm_units=128):
        self.seq_len    = seq_len
        self.embed_dim  = embed_dim
        self.lstm_units = lstm_units
        self.model      = None
        self.word2idx   = {}
        self.idx2word   = {}
        self.vocab_size = 0

    # ── Vocabulary ────────────────────────────────────────────────────────────

    def _build_vocab(self, tokens, min_freq=2):
        """Build word ↔ index mappings; filter rare words."""
        counts = Counter(tokens)
        vocab  = ["<PAD>", "<UNK>"] + [
            w for w, c in counts.most_common() if c >= min_freq
        ]
        self.vocab_size = len(vocab)
        self.word2idx   = {w: i for i, w in enumerate(vocab)}
        self.idx2word   = {i: w for i, w in enumerate(vocab)}
        print(f"  Vocabulary size (min_freq={min_freq}): {self.vocab_size}")

    def _encode(self, tokens):
        unk = self.word2idx["<UNK>"]
        return [self.word2idx.get(t, unk) for t in tokens]

    # ── Data preparation ──────────────────────────────────────────────────────

    def _prepare_sequences(self, tokens):
        """Slide a window of seq_len+1 over the token list."""
        import tensorflow as tf
        encoded = self._encode(tokens)
        X, y = [], []
        for i in range(len(encoded) - self.seq_len):
            X.append(encoded[i : i + self.seq_len])
            y.append(encoded[i + self.seq_len])
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)
        return X, y

    # ── Model ─────────────────────────────────────────────────────────────────

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras import layers

        model = tf.keras.Sequential([
            layers.Embedding(self.vocab_size, self.embed_dim,
                             input_length=self.seq_len),
            layers.LSTM(self.lstm_units, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(self.vocab_size, activation="softmax"),
        ], name="LSTM_NextWord")

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, text_tokens, epochs=30, batch_size=128, val_split=0.1):
        """
        Args:
            text_tokens : Flat list of string tokens.
            epochs      : Training epochs.
            batch_size  : Mini-batch size.
            val_split   : Fraction for validation.
        """
        import tensorflow as tf

        print("[INFO] Building LSTM vocabulary...")
        self._build_vocab(text_tokens)

        print("[INFO] Preparing training sequences...")
        X, y = self._prepare_sequences(text_tokens)
        print(f"  Sequences: {X.shape[0]}  |  Seq length: {self.seq_len}")

        print("[INFO] Building LSTM model...")
        self.model = self._build_model()
        self.model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ]

        print("\n[INFO] Training LSTM next-word predictor...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks,
        )
        return history

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_next(self, context_words, top_k=5, temperature=1.0):
        """
        Predict top-K next words.

        Args:
            context_words : List of preceding words.
            top_k         : Number of suggestions.
            temperature   : Sampling temperature (lower = more deterministic).

        Returns:
            list of (word, probability) tuples.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        unk = self.word2idx["<UNK>"]
        ctx = [self.word2idx.get(w.lower(), unk) for w in context_words]

        # Pad or trim to seq_len
        if len(ctx) < self.seq_len:
            ctx = [0] * (self.seq_len - len(ctx)) + ctx
        else:
            ctx = ctx[-self.seq_len:]

        x    = np.array([ctx])
        preds = self.model.predict(x, verbose=0)[0]

        # Temperature scaling
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))

        top_indices = np.argsort(preds)[::-1][:top_k]
        return [(self.idx2word[i], float(preds[i])) for i in top_indices]

    def generate_text(self, seed_words, num_words=20, temperature=0.8):
        """Generate text by repeatedly calling predict_next."""
        result = list(seed_words)
        for _ in range(num_words):
            preds = self.predict_next(result, top_k=5, temperature=temperature)
            if not preds:
                break
            words  = [p[0] for p in preds]
            probs  = np.array([p[1] for p in preds])
            probs /= probs.sum()
            choice = np.random.choice(words, p=probs)
            result.append(choice)
        return " ".join(result)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_path="lstm_model.keras", vocab_path="vocab.pkl"):
        self.model.save(model_path)
        with open(vocab_path, "wb") as f:
            pickle.dump({
                "word2idx"  : self.word2idx,
                "idx2word"  : self.idx2word,
                "vocab_size": self.vocab_size,
                "seq_len"   : self.seq_len,
            }, f)
        print(f"[INFO] LSTM model saved → {model_path}")

    def load(self, model_path="lstm_model.keras", vocab_path="vocab.pkl"):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)
        with open(vocab_path, "rb") as f:
            data = pickle.load(f)
        self.word2idx  = data["word2idx"]
        self.idx2word  = data["idx2word"]
        self.vocab_size= data["vocab_size"]
        self.seq_len   = data["seq_len"]
        print(f"[INFO] LSTM model loaded ← {model_path}")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED KEYBOARD — SPELL CHECK + NEXT WORD
# ══════════════════════════════════════════════════════════════════════════════

class SmartKeyboard:
    """
    Unified keyboard that combines:
      1. Spell-checker autocorrect (pyspellchecker)
      2. Next-word prediction     (N-gram or LSTM)
    """

    def __init__(self, predictor):
        """
        Args:
            predictor : An NGramPredictor or LSTMNextWordPredictor instance.
        """
        try:
            from spellchecker import SpellChecker
            self.spell = SpellChecker()
        except ImportError:
            self.spell = None
            print("[WARN] pyspellchecker not installed — spell-check disabled.")

        self.predictor = predictor

    def process(self, text, top_k=5):
        """
        1. Spell-check and autocorrect the input text.
        2. Predict next words based on corrected context.
        """
        words = text.strip().split()

        # ── Step 1: Autocorrect ──────────────────
        corrected = []
        corrections = {}
        if self.spell:
            for w in words:
                lower = w.lower()
                if not self.spell.known([lower]):
                    fix = self.spell.correction(lower) or w
                    corrections[w] = fix
                    corrected.append(fix)
                else:
                    corrected.append(w)
        else:
            corrected = words

        corrected_text = " ".join(corrected)

        # ── Step 2: Next-word prediction ─────────
        suggestions = self.predictor.predict_next(corrected, top_k=top_k)

        return {
            "original"      : text,
            "corrected"     : corrected_text,
            "corrections"   : corrections,
            "next_words"    : suggestions,
        }

    def interactive(self):
        print("\n" + "=" * 60)
        print("  SMART KEYBOARD — Autocorrect + Next-Word Prediction")
        print("  Type a sentence fragment. Type 'quit' to exit.")
        print("=" * 60)

        while True:
            try:
                text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue

            result = self.process(text)

            if result["corrections"]:
                print("\n🔧 Autocorrections:")
                for orig, fix in result["corrections"].items():
                    print(f"   {orig} → {fix}")
                print(f"✏️  Corrected: {result['corrected']}")
            else:
                print("✅ No spelling errors.")

            print("\n🔮 Next-word suggestions:")
            for word, prob in result["next_words"]:
                bar = "█" * int(prob * 30)
                print(f"   {word:<15} {bar}  ({prob*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — DEMO
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus():
    """Load a sample corpus from NLTK's Gutenberg dataset."""
    from nltk.corpus import gutenberg
    sents = gutenberg.sents("austen-emma.txt")
    # Flatten to token list and lowercase
    tokens = [w.lower() for sent in sents for w in sent
              if re.match(r"[a-z]", w.lower())]
    sentences = [[w.lower() for w in s if re.match(r"[a-z]", w.lower())]
                 for s in sents]
    print(f"[INFO] Corpus loaded: {len(tokens):,} tokens, {len(sentences):,} sentences")
    return tokens, sentences


if __name__ == "__main__":
    tokens, sentences = load_corpus()

    # ── Demo A: N-gram model ─────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  PART A: N-GRAM NEXT-WORD PREDICTOR")
    print("━" * 60)

    ngram = NGramPredictor(n=3)
    ngram.train(sentences)

    test_contexts = [
        ["she", "was"],
        ["it", "is", "a"],
        ["the", "young", "man"],
        ["i", "do", "not"],
    ]

    for ctx in test_contexts:
        preds = ngram.predict_next(ctx, top_k=5)
        print(f"\nContext : '{' '.join(ctx)}'")
        for word, prob in preds:
            print(f"   {word:<20} ({prob*100:.1f}%)")

    print("\n── Sentence Completion ─────────────────────────────")
    seeds = [["she", "was", "very"], ["i", "am"]]
    for seed in seeds:
        completed = ngram.complete_sentence(seed, max_words=10)
        print(f"  '{completed}'")

    ngram.save("ngram_model.pkl")

    # ── Demo B: LSTM model ───────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  PART B: LSTM NEXT-WORD PREDICTOR")
    print("━" * 60)

    lstm = LSTMNextWordPredictor(seq_len=8, embed_dim=64, lstm_units=128)
    lstm.train(tokens[:50000], epochs=20, batch_size=256)   # use first 50k tokens for speed

    test_contexts_lstm = [
        ["she", "was"],
        ["the", "house", "was"],
        ["he", "looked", "at"],
    ]
    for ctx in test_contexts_lstm:
        preds = lstm.predict_next(ctx, top_k=5)
        print(f"\nContext : '{' '.join(ctx)}'")
        for word, prob in preds:
            print(f"   {word:<20} ({prob*100:.1f}%)")

    print("\n── Text Generation ─────────────────────────────────")
    gen = lstm.generate_text(["she", "smiled"], num_words=15)
    print(f"  '{gen}'")

    lstm.save()

    # ── Smart Keyboard interactive demo (N-gram) ─────────────────────────────
    print("\n" + "━" * 60)
    print("  SMART KEYBOARD DEMO  (N-gram predictor)")
    print("━" * 60)

    kb = SmartKeyboard(predictor=ngram)
    kb.interactive()
