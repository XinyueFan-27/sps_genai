import random
import re
from collections import Counter, defaultdict


def simple_tokenizer(text, frequency_threshold=None):
    """Simple tokenizer that splits text into words."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not frequency_threshold:
        return tokens
    word_counts = Counter(tokens)
    return [t for t in tokens if word_counts[t] >= frequency_threshold]


def analyze_bigrams(text, frequency_threshold=None):
    """Compute bigram probabilities: p(w2 | w1) = C(w1, w2) / C(w1)."""
    words = simple_tokenizer(text, frequency_threshold)
    bigrams = list(zip(words[:-1], words[1:]))

    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(words)

    bigram_probs = defaultdict(dict)
    for (word1, word2), count in bigram_counts.items():
        bigram_probs[word1][word2] = count / unigram_counts[word1]

    return list(unigram_counts.keys()), bigram_probs


class BigramModel:
    def __init__(self, corpus):
        # main.py 传入的是句子列表，合并成一个大文本再统计
        text = " ".join(corpus)
        self.vocab, self.bigram_probs = analyze_bigrams(text)

    def generate_text(self, start_word, num_words=20):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(num_words - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:  # 没有后续词就停止
                break
            next_word = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)