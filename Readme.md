# An Empirical Study of User Playback Interactions and Engagement in Mobile Video Viewing

<span style="font-size:24px; font-weight:bold">[Read full paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10982068&tag=1)</span>

## Overview

<p>
When using online video platforms like YouTube and Netflix, users often adjust playback speed, skip forward, or rewind. In this study, we developed a mobile app to collect real-world viewing logs and satisfaction responses and conducted a field study with 25 participants. Our findings reveal that specific playback behaviors, such as scrubbing and backward skipping, are closely associated with user engagement patterns. Importantly, we show that video abandonment does not always signal dissatisfaction. These insights provide valuable guidance for designing smarter recommendation algorithms and improving user experiences in online video streaming services.
</p>

## Repository Structure

```bash
├── Analysis/ # Notebooks presenting table results and analyses from the paper
│   ├── util.py # Data analysis utility functions
│   ├── table2.ipynb
│   ├── table3.ipynb
│   ├── table4.ipynb
│   ├── table5.ipynb
│   ├── table6.ipynb
│   └── category.ipynb

├── Figure/ # Figure files attached in the paper
│   ├── App(Figure 1).jpg
│   ├── DataCount(Figure 2).pdf
│   ├── AbandonmentReason(Figure 3).pdf
│   └── figure.ipynb

├── requirements.txt
├── preProcess.xlsx # Preprocessed data
└── rawData.xlsx # Raw data (original)
```

<p>
To protect participant privacy, sensitive data (e.g., ESM responses) have been masked (e.g., replaced with ##########) in the Excel file.
</p>

## Installation

To set up the environment and run the notebooks:

### 1. Clone this repository

```bash
git clone https://github.com/eunnho98/YouTubeESM
```

### 2. (Optional) Create and activate a virtual environment

It is recommended to use a virtual environment with Python 3.9.21.

Make sure Python 3.9.21 is installed on your system. You can check with:

```bash
python3 --version
```

<br>

Then create and activate a virtual environment:

```bash
# Create a virtual environment using Python 3.9.21
python3.9 -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 3. Install required packages

Install the Python dependencies listed in _requirements.txt_:

```bash
pip install -r requirements.txt
```

After installation, you can run the notebooks in the Analysis/ or Figure/ directories.

## Citation

_E. Kim, S. Oh and S. Park_, **An Empirical Study of User Playback Interactions and Engagement in Mobile Video Viewing**, in IEEE Access.

```
@article {kim2025empirical,
  author = {Kim, Eunnho and Oh, Seungjae and Park, Sangkeun},
  journal = {IEEE Access},
  title = {An Empirical Study of User Playback Interactions and Engagement in Mobile Video Viewing},
  year = {2025},
  volume = {13},
  pages = {78272-78289},
  doi = {10.1109/ACCESS.2025.3566402}
}
```
