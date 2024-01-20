# MindMap AI: Emotion Classification in Relation to Mental Health 

Welcome to our project repository for CSCI-567, the Machine Learning graduate course at USC. This repository features advanced models for facial emotion classification in images and emotion classification in text. Dive into our [final report](final_project_report.pdf) for a detailed exploration of our project, including experiments and results.
> **Note:** For tracking experiments, we use Comet ML. Follow the instructions [here](https://www.comet.com/docs/v2/guides/getting-started/quickstart/) to get an API key. If not using Comet ML, ignore related code sections.

## Repository Overview:

### Image Emotion Classifier
- **Folders:**
  - `image-emotion-classifier`: CNN-based models for image emotion classification.
    - `cnn-architecture`: Custom CNN models trained on the fer2013 dataset.
    - `resnet50`: Variations of ResNet50 with custom layers, also trained on fer2013.
    - `vgg`: VGG16 and VGG19 models with fine-tuning and custom layers. Specific files include `vgg16_v1.ipynb`, `vgg16_v1_mtcnn.ipynb`, and `vgg19_fer.ipynb`.

### Text Emotion Classifier
- **Folders:**
  - `text-emotion-classifier`: Transformer-based models for text emotion classification.
    - `cleaning_augmentation`: Data cleaning and augmentation code for text dataset.
    - `roberta-base`: Our high-performing RoBERTa model with complete training and testing code.
    - `hybrid_models`: Combining Bi-LSTM with attention layers and transformer models.
      - `bi-lstm_attention_roberta` and `bi-lstm_attention_distilbert`.
    - `alternate_dataset_checkpoint_model`: Finetuning models with the [Self-Reported Emotion](https://github.com/EmotionDetection/Self-Reported-SR-emotion-dataset) dataset.
> **Note**: All models besides `alternate_dataset_checkpoint_model` use the [GoEmotions](https://huggingface.co/datasets/go_emotions) dataset.

## Getting Started with the Classifiers:

### Image Classifier
- **Setup:**
  1. Install Anaconda or create a virtual environment.
  2. Install dependencies: `pip install -r requirements.txt` (upgrade pip beforehand).
  3. Download the [required dataset](https://drive.google.com/file/d/1uyCOBCyoVyBsKcC5df26_qfC8roIatd3/view?usp=sharing) and adjust file paths in the code.
  4. Set up a `.env` file with Comet ML credentials (optional).
  > **Note**: To reproduce `vgg16_v1_mtcnn.ipynb`, first download the mtcnn fer 2013 dataset from the link above. Then, execute all cells in the notebook, except those containing the preprocess_images function.

### Text Classifier
- **Google Colab Usage:**
  - Copy the code from `.ipynb` files in the `text-emotion-classifier` folder to a new Google Colab notebook to run code directly.
- **Local Setup:**
  1. Create a virtual environment.
  2. Install dependencies: `pip install -r requirements.txt` (upgrade pip first).
  3. Run `main.py`.
  > **Note**: `cleaning_augmentation` is the only code that needs to be run locally. The rest of the code can be run on Google Colab.

## Workflow and Contribution Guidelines:

- **Before Starting:**
  - Discuss work items in team meetings.
  - Ensure there's a GitHub issue for the task you're taking on.

- **Starting Work:**
  - Assign yourself to the issue on GitHub.
  - Sync the latest changes from the `main` branch.
  - Create and switch to a new branch for your work.

- **Development Process:**
  - Commit and push changes frequently, adhering to [Conventional Commits](https://www.conventionalcommits.org/).
  - Start a pull request early, marking it as a draft.
  - Write and run unit tests (if applicable).
  - Resolve merge conflicts promptly.

- **Finishing Up:**
  - Clean and review your code.
  - Ensure all tests pass.
  - Finalize the pull request and request code reviews.
  - Implement review feedbacks and merge upon approval.
