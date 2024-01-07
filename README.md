# Plagiarism Checker with CNN

## Overview

This project implements a plagiarism checker using Convolutional Neural Networks (CNN). It is designed to analyze pairs of questions and predict whether they are duplicate or not. The CNN model is trained on the Quora Question Pair dataset from Kaggle.

## Table of Contents

- [Data](#data)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Data

Please download the Quora Question Pair dataset from Kaggle and place it in the 'data' folder within your project directory. The dataset can be found [here](https://www.kaggle.com/c/quora-question-pairs/data).

## Project Structure

```plaintext
- /plagiarism_checker
  - /data
    - quora_question_pair_dataset.csv
  - /src
    - clean.py
    - model.py
    - utility.py
    - main.py
  - README.md
  - requirements.txt
```

## Requirements

Make sure you have Python installed. You can download it from the [official Python website](https://www.python.org/downloads/).

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/vikasharma005/plag_checker.git
   ```

2. Navigate to the project directory:

   ```bash
   cd plag_checker
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the Quora Question Pair dataset from Kaggle and place it in the 'data' folder.

## Usage

Run the plagiarism checker using the following command:

```bash
python main.py
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Word2Vec model: GoogleNews-vectors-negative300.bin.gz
- Quora Question Pair dataset: Kaggle
