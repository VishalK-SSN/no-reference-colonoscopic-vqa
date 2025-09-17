import os
import pandas as pd


def extract_video_sort_key(video_name):
    """Extract sort key from video filename (e.g., '1-14.mp4' -> (1, 14))"""
    base_name = str(video_name).replace('.mp4', '')
    parts = base_name.split('-')
    try:
        return tuple(map(int, parts))
    except (ValueError, AttributeError):
        return (base_name,)


def convert_excel_to_sorted_csv(excel_file, sheet_name, output_csv):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # Sort by the first column (assuming it contains video names)
    video_col = df.columns[0]
    df[video_col] = df[video_col].astype(str)
    df_sorted = df.sort_values(by=video_col, key=lambda col: col.map(extract_video_sort_key))
    
    # Save to CSV
    df_sorted.to_csv(output_csv, index=False)
    print(f"Sheet '{sheet_name}' sorted and saved as {output_csv}")


def main():
    """Main function to process the SVD features file"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'objective-1', 'features'))
    
    input_file = os.path.join(base_dir, 'original', 'extra-features.xlsx')
    output_dir = os.path.join(base_dir, 'cleaned')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'cleaned-extra-features.csv')
    sheet_name = 'Sheet1'
    
    convert_excel_to_sorted_csv(input_file, sheet_name, output_file)


if __name__ == "__main__":
    main()