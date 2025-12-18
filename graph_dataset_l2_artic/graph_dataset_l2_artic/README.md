readme_text = """
📄 README.txt - L2 Arctic Phoneme Dataset Preparation
-----------------------------------------------------

🔤 Original Phoneme Inventory
The original dataset contains 40 standard phonemes (ARPA format):

['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY',
 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P',
 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
 'V', 'W', 'Y', 'Z', 'ZH']

🌍 Phoneme Nationality Tagging
Each speaker ID ends with a character representing their L1 nationality:

nationalities = {
    "A": "Arabic",
    "C": "Chinese",
    "I": "Indian",
    "K": "Korean",
    "S": "Spanish",
    "V": "Vietnamese"
}

In processing, we use this character as a suffix for each phoneme (e.g., AA_V for a Vietnamese speaker).

📁 Input Files
The following CSV files are used:

- L2_arctic_train.csv
- dev.csv
- test.csv

Each file contains rows with:
- Path
- Canonical (gold phoneme sequence)
- Transcript (spoken phoneme sequence)

🧠 Preprocessing Logic
We define a function `process_record(record)` that:

1. Extracts speaker ID from the Path using:
   r"L2_arctic_WAV/([A-Z]+)_arctic_"
2. Adds phoneme suffix based on the speaker’s L1 (speaker_id[-1]):
   - For both Canonical and Transcript
   - Skips phonemes containing * (e.g., aa* → ignored)
   - Appends _X (e.g., AA → AA_V)
3. Assigns records to:
   - speaker_dfs[speaker_id] if the speaker is in the Test-Dev set
   - train_records otherwise

🧪 Test-Dev Speaker Set
Speakers used for dev/test splitting:
["EBVS", "THV", "TNI", "BWC", "YDCK", "YBAA"]
Each of their records is split 50/50 between dev and test.

📊 Output Summary

| File       | Records | Description                          |
|------------|---------|--------------------------------------|
| train.csv  | 2700    | General training data                |
| dev.csv    | 449     | Development set (from 6 speakers)    |
| test.csv   | 450     | Test set (from same 6 speakers)      |

🔣 Phoneme Statistics
- Total Unique Phonemes after processing: 240
  - This includes base phonemes + nationality suffixes (e.g., AA_V, S_K, etc.)
  - *-marked phonemes are excluded
"""

# Write to file
with open("README.txt", "w") as f:
    f.write(readme_text)

"README.txt file has been written successfully."