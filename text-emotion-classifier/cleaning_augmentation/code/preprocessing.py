import numpy as np
import nltk, random, re
from nltk.corpus import wordnet
from emoji import demojize
from model_config import get_parameters
from helper_functions import get_wordnet_pos

class Preprocess:
    def __init__(self):
        self.settings = get_parameters()
        self.tokenizer = self.settings['tokenizer']
        self.allowed_chars = re.compile(self.settings['reg_exps']['allowable_chars'])
        self.brackets_stars_pattern = re.compile(self.settings['reg_exps']['brackets_stars_pattern'])
        self.space_before_punc = re.compile(self.settings['reg_exps']['space_before_punc'])

    # tokenizes and formats data into format accepted by distilbert model
    def tokenize_data(self, batch):
        text = batch['text']
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        labels_batch = {k: batch[k] for k in batch.keys() if k in self.settings['emotions']}
        labels_matrix = np.zeros((len(text), self.settings['num_labels']))

        for idx, label in enumerate(self.settings['emotions']):
            labels_matrix[:, idx] = labels_batch[label]

        encoding['labels'] = labels_matrix.tolist()

        return encoding

    # converts emojis to text
    def convert_emojis_to_text(self, batch):
        text = batch['text']

        for index in range(len(text)):
            text[index] = demojize(text[index])

        batch['text'] = text
        return batch
    
    # applies a series of funcitons to clean the data
    def clean_data(self, batch):
        clean_batch = self._clean_dirty_instances(batch)
        # try correcting typos
        # try removing links or email addresses
        # try removing or replacing numbers
        # expand contradicitons (don't to do not)
        return clean_batch
    
    # augments data for underrepresented labels using synonym replacement
    def augment_data(self, batch):
        original_text_length = len(batch['text'])
        text = batch['text']
        ids = batch['id']

        for emotion in self.settings['emotions_to_augment']:
            for index, value in enumerate(batch[emotion]):
                if value and index < original_text_length:
                    augmentation_factor = 4 if emotion in self.settings['emotions_to_augment_heavy'] else 1

                    for _ in range(augmentation_factor):
                        new_sentence, replaced = self._synonym_replacement(text[index])

                        if replaced:
                            text.append(new_sentence)
                            new_id = ids[index] + "_aug" 
                            ids.append(new_id)

                            for inner_emotion in self.settings['emotions']:
                                if inner_emotion == emotion:
                                    batch[inner_emotion].append(1)
                                else:
                                    batch[inner_emotion].append(0)

        batch['text'] = text
        batch['id'] = ids
        return batch
    
    # cleans 'dirty' instances which have special chars not contributing to emotion
    def _clean_dirty_instances(self, batch):
        text = batch['text']

        for index in range(len(text)):
            if not self.allowed_chars.fullmatch(text[index]):
                text[index] = re.sub(self.brackets_stars_pattern, ' ', text[index])
                allowed_sequences = self.allowed_chars.findall(text[index])
                text[index] = ''.join(allowed_sequences)
                text[index] = re.sub(r'\s+', ' ', text[index]).strip()
                text[index] = re.sub(self.space_before_punc, r'\1', text[index]).strip()

        batch['text'] = text
        return batch
              
    # conducts synonym replacement, creating new sentence
    def _synonym_replacement(self, sentence):
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        new_sentence = sentence
        replaced = False

        replaceable_words = [(word, get_wordnet_pos(pos)) for word, pos in pos_tags if get_wordnet_pos(pos) is not None]
        if not replaceable_words:
            return new_sentence, replaced

        word_to_replace, pos_tag = random.choice(replaceable_words)
        synonyms = [syn.name().split('.')[0] for syn in wordnet.synsets(word_to_replace, pos=pos_tag) if syn.name().split('.')[0] != word_to_replace]

        if synonyms:
            synonym = random.choice(synonyms)
            new_sentence = new_sentence.replace(word_to_replace, synonym, 1)
            replaced = True

        return new_sentence, replaced