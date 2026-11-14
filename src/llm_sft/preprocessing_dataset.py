import pandas as pd
import os

def preprocess_snap_for_autotrain(input_csv_path, output_csv_path):
    """
    Preprocess S_NAP.csv dataset for HuggingFace Autotrain format.
    
    Args:
        input_csv_path (str): Path to the S_NAP.csv file
        output_csv_path (str): Path where the processed CSV will be saved
    """
    print("Loading S_NAP.csv dataset...")
    chunk_size = 10000
    processed_data = []
    
    for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size):
        print(f"Processing chunk with {len(chunk)} rows...")
        for _, row in chunk.iterrows():
            prefix = str(row['prefix'])
            next_activity = str(row['next'])
            unique_activities_str = str(row['unique_activities'])
            try:
                if unique_activities_str.startswith('{') and unique_activities_str.endswith('}'):
                    activities_content = unique_activities_str[1:-1]
                    if activities_content.strip():
                        activities_list = [activity.strip().strip("'\"") for activity in activities_content.split("', '")]
                        if len(activities_list) == 1 and "', '" not in activities_content:
                            activities_list = [activity.strip().strip("'\"") for activity in activities_content.split(", ")]
                    else:
                        activities_list = []
                else:
                    activities_list = [activity.strip().strip("'\"") for activity in unique_activities_str.split(",")]
                if "[END]" not in activities_list:
                    activities_list.append("[END]")
                unique_activities = "{" + ", ".join([f"'{activity}'" for activity in sorted(activities_list)]) + "}"
                
            except Exception as e:
                if "[END]" not in unique_activities_str:
                    if unique_activities_str.endswith('}'):
                        unique_activities = unique_activities_str[:-1] + ", '[END]'}"
                    else:
                        unique_activities = unique_activities_str + ", [END]"
                else:
                    unique_activities = unique_activities_str
            
            human_text = f"Given the process prefix: {prefix}. The allowed unique activity names are: {unique_activities}. What is the next activity?"
            assistant_text = next_activity
            formatted_text = f"### Human: {human_text} ### Assistant: {assistant_text}"
            processed_data.append({"text": formatted_text})
    
    print(f"Processed {len(processed_data)} examples total.")
    df_output = pd.DataFrame(processed_data)
    df_output.to_csv(output_csv_path, index=False)
    print(f"Dataset saved to: {output_csv_path}")
    print(f"Final dataset shape: {df_output.shape}")
    
    print("\nFirst 3 examples:")
    for i in range(min(3, len(df_output))):
        print(f"\nExample {i+1}:")
        print(df_output.iloc[i]['text'])

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(current_dir, "S_NAP.csv")
    output_csv = os.path.join(current_dir, "S_NAP_autotrain.csv")
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} not found!")
        return
    
    preprocess_snap_for_autotrain(input_csv, output_csv)

if __name__ == "__main__":
    main()