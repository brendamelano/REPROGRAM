import pandas as pd

def extract_crispr_guides(file_path):
    """
    Extracts gene names and guide sequences from a CRISPR guide file.

    Args:
    file_path (str): Path to the CRISPR guide file.

    Returns:
    pd.DataFrame: A DataFrame containing 'Gene Name' and 'Guide Sequence'.
    """
    # Initialize lists to store gene names and guide sequences
    gene_names = []
    guide_sequences = []

    # Open the file and process each line
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        
        for line in file:
            # Split each line into columns based on tab-delimitation
            columns = line.strip().split("\t")
            # Extract gene name and guide sequence columns (adjust indices if needed)
            gene_names.append(columns[4])  # Assuming gene symbol is the 5th column
            guide_sequences.append(columns[18])  # Assuming guide sequence is the 18th column

    # Create a DataFrame from the extracted data
    guides_df = pd.DataFrame({
        "Gene Name": gene_names,
        "Guide Sequence": guide_sequences
    })

    return guides_df