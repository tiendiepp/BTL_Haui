import pandas as pd

def load_and_prepare_data(file_path):
    cell_df = pd.read_csv(file_path)
    cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
    cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
    return cell_df
