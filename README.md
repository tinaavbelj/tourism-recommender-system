# Tourism Recommender System

## Installation

### Clone git repository

`git clone https://github.com/tinaavbelj/tourism-recommender-system.git`

### Install Poetry

`curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python`

### Create virtual environment & install dependencies

```
python -m venv venv
source venv/bin/activate
poetry install
```

## Config

Download `data.zip` from Google Drive and extract its contents into root directory.

## Run

Run `jupyter lab` from the root directory.

### Object Selection (DFMF)

To run object selection navigate to `object-selection/jupyter`.

### Classification

To run classification navigate to `classification/jupyter` (images) or `classification/jupyter-text` (text).
