# Emotion Classification in Relation to Mental Health 

Welcome to the repository for our group project for CSCI-567, USC's Machine Learning graduate course. This repo contains a variety of transformer-based models for emotion classification in text and convolutional neural network (CNN) models tailored for facial emotion classification in images.

## Navigating the Repo:
- This repo has two main folders: `image-emotion-classification` and `text-emotion-classification`. Each folder contains the code for the respective model type. `image-emotion-classification` holds our CNN-based models, and `text-emotion-classification` holds our transformer-based models. Within each folder are a variety of subfolders or ipynb files containing the code for the different models we trained. Below is a list describing the models in each of the main folders:
### image-emotion-classification:
- `cnn-architecture`: [Blank]
- `resnet50`: [Blank]
- `vgg16`: [Blank]

### text-emotion-classification:
- `cleaning_augmentation`: This folder contains the code for the data cleaning and augmentation process we used for the text data before training a DistilBERT model.
- `roberta-base`: This ipynb file contains the pre-ran code for training and testing a DistilBERT model on the text data. This is our highest-performing transformer model. It shows the full preprocessing, training, testing, and result visualization of a DistilBERT model for our task. 
- [hybrid and any other models]

## How to Install Python Packages:
- For code using a `requirements.txt` file, run the following in a command prompt in the directory where you cloned this repo:
```bash
pip install -r requirements.txt
```
> Note: Upgrade 'pip' before installing the packages.

## Tracking Results:
- We are using Comet ML to track our experiments. If you want to track model settings and training/testing results, create an account on Comet ML and get your API key. You can find instructions on how to do that [here](https://www.comet.com/docs/v2/guides/getting-started/quickstart/).
### For Windows Users:
- Once you have your API key, run the following commands in a stand-alone PowerShell in the directory where you cloned this repo:
```bash
$env:COMET_API_KEY="<your-api-key>"
$env:COMET_PROJECT_NAME="<your-project-name>"
```
- You can now run the code, and it will automatically log your experiments to your project on your Comet account.
> Note: Some code might directly prompt you to enter your API key. In that case, you can skip the above steps and just enter your API key when prompted.

## Workflow

### Before Getting Started

- Discuss what you'll be working on in team meetings
- Make sure there is a GitHub issue in the repo that is tracking the item you're working on

### Get Started on a Work Item

- On GitHub
  - Assign yourself to the issue you'll work on (if not already)

- In your local Git repo
  - Switch to `main` branch (unless otherwise specified)
  - Sync latest changes on `main` branch (IMPORTANT!!!)
  - Create and checkout a new branch from `main`
    - Branch name may start with the issue ID followed by feature, like `10-code-review-doc`
  - Start coding

### Work on an Item

- Commit and push changes frequently
  - Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
  - Check out the reason behind [Adopting Conventional Commits](/blog/memo-2021-07-21#adopting-conventional-commits)

- Create a pull request on GitHub as early as you make the first commit
  - Title should start with `Draft:` and followed by the feature, like `Draft: Code Review Doc`
  - Description should start with a line like `Closes #10` to let GitHub link the issue with the pull request automatically

- Write unit tests while developing new features (if applicable)

- Try to resolve merge conflicts (if any) with the target branch as you notice

### Finish up

- Clean up code
- Run and pass all related unit tests (if applicable)
- Remove `Draft:` from title of the pull request
- Request a code review
- Wait for review
- Implement feedbacks (if any) by continue adding commits and request another review
- Wait for approval and merge

