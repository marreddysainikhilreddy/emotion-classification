from transformers import TrainingArguments, AutoTokenizer, \
                         AutoModelForSequenceClassification

def get_parameters():

    columns_to_remove = [
        'author',
        'subreddit',
        'link_id',
        'parent_id',
        'created_utc',
        'rater_id',
        'example_very_unclear',
        'admiration',
        'approval',
        'caring',
        'confusion',
        'desire',
        'disappointment',
        'disapproval',
        'excitement',
        'nervousness',
        'pride',
        'realization',
        'relief',
        'remorse',
        'surprise'
    ]

    emotions = [
        'amusement',
        'anger',
        'annoyance',
        'curiosity',
        'disgust',
        'embarrassment',
        'fear',
        'gratitude',
        'grief',
        'joy',
        'love',
        'optimism',
        'sadness',
        'neutral'
    ]

    emotions_to_augment = [
        'embarrassment',
        'fear',
        'grief',
        'anger',
        'disgust',
        'joy',
        'sadness'
    ]
    
    emotions_to_augment_moderate = [
        'anger',
        'disgust',
        'joy',
        'sadness'
    ]
    
    emotions_to_augment_heavy = [
        'embarrassment',
        'fear',
        'grief'
    ]

    training_args = TrainingArguments(
        output_dir='./model_output',
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        report_to='comet_ml'
   )

    regular_expressions = {
        'allowable_chars': r'[A-Za-z0-9\s!?,\'\".“”’-]+', # matches a sequence of one or more of the following characters: (A-Z) or (a-z) letter; (0-9) digits; & specific punctuation
        'brackets_stars_pattern': r'\[.*?\]|\*\*.*?\*\*', # matches any sequence enclosed in brackets or double stars
        'space_before_punc': r'\s+([!?,\'\".\-])' # matches unnecessary spaces before punctuation
    }

    id2label = {idx:label for idx, label in enumerate(emotions)}
    label2id = {label:idx for idx, label in enumerate(emotions)}

    settings = {
        'tokenizer': AutoTokenizer.from_pretrained('distilbert-base-uncased'),
        'columns_to_remove': columns_to_remove,
        'emotions': emotions,
        'emotions_to_augment': emotions_to_augment,
        'emotions_to_augment_moderate': emotions_to_augment_moderate,
        'emotions_to_augment_heavy': emotions_to_augment_heavy,
        'training_args': training_args,
        'reg_exps': regular_expressions,
        'replacement_rate': 0.1,
        'num_labels': len(emotions),
        'tokenizer': AutoTokenizer.from_pretrained("distilbert-base-uncased"), #, use_fast=True)
        'model': AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(emotions),
                                                           id2label=id2label,
                                                           label2id=label2id)
    }

    return settings