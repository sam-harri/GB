import pandas as pd

def generate_overlapping_sequences(data, sequence_length=10, overlap=5):
    """
    Generate overlapping sequences from a pandas DataFrame.

    Parameters:
    - data: pandas DataFrame containing the data.
    - sequence_length: length of each sequence.
    - overlap: number of overlapping points in two consecutive sequences.

    Returns:
    - A list of sequences.
    """
    sequences = []
    for i in range(0, len(data) - sequence_length + 1, sequence_length - overlap):
        sequences.append(data.iloc[i: i + sequence_length].values)
    return sequences

# Load data from CSV
data = pd.read_csv("your_file.csv")

# Generate overlapping sequences
sequence_length = 20  # Length of each sequence
overlap = 10  # Number of overlapping data points in two consecutive sequences
sequences = generate_overlapping_sequences(data, sequence_length, overlap)

# Now, `sequences` contains overlapping sequences of length `sequence_length`
