"""
Load transcripts and split by SPEAKER, not by row, to avoid leaking
the same speaker's language style into both train and test.

Expected input CSV (config.RAW_CSV) with columns:
    speaker_id, text, label   (label: 0 = control, 1 = dementia)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_CSV)
    required = {"speaker_id", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"transcripts.csv is missing columns: {missing}")
    return df


def speaker_level_split(df: pd.DataFrame):
    """
    Split so that every transcript from a given speaker ends up in only
    one of train/test. Stratifies on the speaker's majority label.
    """
    speaker_labels = (
        df.groupby("speaker_id")["label"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )

    train_speakers, test_speakers = train_test_split(
        speaker_labels["speaker_id"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=speaker_labels["label"],
    )

    train_df = df[df["speaker_id"].isin(train_speakers)].reset_index(drop=True)
    test_df = df[df["speaker_id"].isin(test_speakers)].reset_index(drop=True)

    # Sanity check: no speaker overlap
    overlap = set(train_df["speaker_id"]) & set(test_df["speaker_id"])
    assert not overlap, f"Speaker leakage detected: {overlap}"

    return train_df, test_df


if __name__ == "__main__":
    df = load_raw_data()
    train_df, test_df = speaker_level_split(df)
    print(f"Total transcripts: {len(df)}  |  Speakers: {df['speaker_id'].nunique()}")
    print(f"Train: {len(train_df)} transcripts, {train_df['speaker_id'].nunique()} speakers")
    print(f"Test:  {len(test_df)} transcripts, {test_df['speaker_id'].nunique()} speakers")
    print("Train class balance:\n", train_df["label"].value_counts(normalize=True))
    print("Test class balance:\n", test_df["label"].value_counts(normalize=True))
