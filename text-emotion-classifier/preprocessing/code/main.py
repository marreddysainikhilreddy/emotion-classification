import numpy as np
import nltk
from transformers import Trainer
from transformers.integrations import CometCallback
from datasets import load_dataset, DatasetDict
from model_config import get_parameters
from helper_functions import compute_metrics
from preprocessing import Preprocess

def main():
   # initialize objects
   settings = get_parameters()
   preprocess = Preprocess()

   print("\nLoading & preprocessing dataset...")
   # load GoEmotions dataset & create train, test, validation splits
   train_test = load_dataset('go_emotions', name='raw', split='train').train_test_split(test_size = 0.2)
   test_validation = train_test['test'].train_test_split(test_size=0.5)
   go_emotions = DatasetDict({
      'train': train_test['train'],
      'test': test_validation['test'],
      'validation': test_validation['train']})

   # remove columns containing non-target emotions & instances containing no positive target emotions
   print("\nRemoving non-target emotion columns & instances containing no positive classifications...")
   go_emotions = go_emotions.remove_columns(settings['columns_to_remove'])
   go_emotions = go_emotions.filter(lambda row: any(row[emotion] == 1 for emotion in settings['emotions']))
   
   tokenizer = settings['tokenizer']
   def tokenize_function(examples):
      # Tokenize the texts and return the full tokenizer output
      return tokenizer(examples["text"], truncation=True, padding=False)

   # Apply the tokenize function to all the splits in dataset
   tokenized_dataset = go_emotions.map(tokenize_function, batched=True)

   lengths = [len(ids) for ids in tokenized_dataset['train']['input_ids']]

   # Calculate descriptive statistics
   lengths = np.array(lengths)
   mean_length = np.mean(lengths)
   median_length = np.median(lengths)
   max_length = np.max(lengths)

   # Determine a suitable padding length
   padding_length_95th = np.percentile(lengths, 95)
   padding_length_99th = np.percentile(lengths, 99)

   print(f"Mean Length: {mean_length}")
   print(f"Median Length: {median_length}")
   print(f"Max Length: {max_length}")
   print(f"95th Percentile Length: {padding_length_95th}")
   print(f"99th Percentile Length: {padding_length_99th}")

   import matplotlib.pyplot as plt

   plt.hist(lengths, bins=30)
   plt.title('Distribution of Text Lengths')
   plt.xlabel('Length')
   plt.ylabel('Number of Texts')
   plt.show()

   # 'clean' dataset
   print("\nCleaning dataset...")
   go_emotions = go_emotions.map(preprocess.clean_data, batched=True)
   
   # augment data for underrepresented labels
   # download nltk packages for data augmentation
   print("\nDownloading nltk packages for data augmentation...")
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
  
   print("\nAugmenting low frequency labels...")
   go_emotions = go_emotions.map(preprocess.augment_data, batched=True)
   
   # 'clean' dataset
   print("\nCleaning dataset...")
   go_emotions = go_emotions.map(preprocess.clean_data, batched=True)
   
   # convert emojis to text
   print("\nConverting emojis to text...")
   go_emotions = go_emotions.map(preprocess.convert_emojis_to_text, batched=True)

   # tokenize text for input into distilbert & set format
   print("\nTokenizing dataset...")
   tokenized_dataset = go_emotions.map(preprocess.tokenize_data, batched=True, remove_columns=go_emotions['train'].column_names)
   tokenized_dataset.set_format("torch")

   print("Data Preprocessing Complete.")

   # define trainer
   trainer = Trainer(
      model=settings['model'],
      args=settings['training_args'],
      train_dataset=tokenized_dataset['train'],
      eval_dataset=tokenized_dataset['validation'],
      tokenizer=settings['tokenizer'],
      compute_metrics=compute_metrics,
      callbacks=[CometCallback()]
   )

   # train and evaluate model
   print("\nTraining model...")
   train_result = trainer.train()
   print(f"Training completed.\nResults: {train_result}")

   print("\nEvaluating model on validation dataset...")
   eval_result = trainer.evaluate()
   print(f"Validation evaluation completed.\nResults: {eval_result}")

if __name__ == "__main__":
    main()