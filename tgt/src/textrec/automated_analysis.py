from collections import Counter

import numpy as np
import spacy
import wordfreq

from textrec.paths import paths

from .util import mem

# Computed by compute_gating_threshold. See also GatedCapTask.js
GATING_THRESHOLD = -0.989417552947998

spacy_nlp = None
def get_spacy():
    global spacy_nlp
    if spacy_nlp is None:
        print("Loading SpaCy...", end='', flush=True)
        spacy_nlp = spacy.load('en_core_web_md')
        print("Hacking SpaCy...")
        spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'), before='parser')
        print("done")
    return spacy_nlp


@mem.cache(verbose=0)
def pos_counts(text):
    nlp = get_spacy()
    return Counter(token.pos_ for token in nlp(text))

@mem.cache
def wordfreqs(text):
    freqs = []
    for tok in wordfreq.tokenize(text, 'en'):
        freq = wordfreq.zipf_frequency(tok, 'en')
        if freq != 0:
            freqs.append(freq)
    return np.array(freqs)


def mean_log_freq(text):
    return np.mean(wordfreqs(text))


def total_rarity(text):
    rarities = 1 - wordfreqs(text) / 7.
    return np.sum(rarities)


@mem.cache
def eval_logprobs_unconditional(text):
    from . import onmt_model_2

    wrapper = onmt_model_2.models['coco_lm']
    tokens = onmt_model_2.tokenize(text)
    logprobs = wrapper.eval_logprobs('.', tokens, use_eos=True)
    return np.mean(logprobs)

@mem.cache
def taps_to_type(txt, threshold=None):
    from . import onmt_model_2

    def rec_gen(context, prefix=None):
        return onmt_model_2.get_recs('coco_lm', '.', context, prefix=prefix)

    recs_log = []
    actions = []
    # Invariant: performing [actions] types txt[:idx]
    idx = 0
    while idx < len(txt):
        sofar = txt[:idx]
        if ' ' in sofar:
            last_space_idx = sofar.rindex(' ')
        else:
            last_space_idx = -1
        prefix = sofar[:last_space_idx + 1]
        cur_word = sofar[last_space_idx + 1:]
        cur_desired_word = txt[last_space_idx + 1:].split(' ', 1)[0]
#         if cur_desired_word[-1] in ',.;-':
#             cur_desired_word = cur_desired_word[:-1]
#         print(repr(prefix), repr(cur_word), repr(cur_desired_word))
        recs = rec_gen(onmt_model_2.tokenize(prefix), prefix=cur_word)
        potentially_suggested_words = [word for word, rec in recs]
        suggested_words = potentially_suggested_words
        max_prob = max(prob for word, prob in recs if prob is not None)
        if threshold is not None:
            show_recs = max_prob > threshold
            if not show_recs:
                suggested_words = []
        if cur_desired_word in suggested_words:
            action = dict(type='rec', which=suggested_words.index(cur_desired_word), word=cur_desired_word, cur_word=cur_word)
            idx = last_space_idx + 1 + len(cur_desired_word) + 1
        else:
            action = dict(type='key', key=txt[idx], cur_word=cur_word)
            idx += 1
        action['recs_shown'] = suggested_words
        action['recs_all'] = potentially_suggested_words
        action['max_rec_prob'] = max_prob
        actions.append(action)
    return actions


def depunct(text):
    return text.replace('.', '').replace(',', '')

def all_taps_to_type(stimulus, text, prefix):
    # taps_to_type is broken wrt word-ending punctuation. Hack around that.
    text_without_punct = depunct(text)
    num_punct = len(text) - len(text_without_punct)
    taps_by_cond = dict(
        norecs=[dict(type='key', key=c) for c in text_without_punct],
        standard=taps_to_type(None, text_without_punct),
        gated=taps_to_type(None, text_without_punct, threshold=GATING_THRESHOLD),
    )
    res = {}
    for condition, taps in taps_by_cond.items():
        res[f'{prefix}tapstotype_{condition}'] = len(taps) + num_punct
        res[f'{prefix}idealrecuse_{condition}'] = len([action for action in taps if action['type'] == 'rec'])
        if condition != 'norecs':
            beginning_of_word_actions = [action for action in taps if action['cur_word'] == '']
            res[f'{prefix}bow_recs_offered_{condition}'] = len([action for action in beginning_of_word_actions if action['recs_shown']])
            res[f'{prefix}bow_recs_idealuse_{condition}'] = len([action for action in beginning_of_word_actions if action['type'] == 'rec'])
    return res
