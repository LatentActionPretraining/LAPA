import json
import pandas as pd
import argparse
import json


def assign_bin(data_point, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= data_point < bins[i + 1]:
            return i
    if data_point >= bins[len(bins)-1]:
        return len(bins) - 2
    if data_point <= bins[0]:
        return 0
    return None  # For data points outside the bin range

def main(args):
    input_jsonl = []
    output_jsonl = [] 

    # read json file not jsonl
    with open(args.input_path, 'r') as infile:
        input_jsonl = json.load(infile)

    # Extracting action lists
    pos_x_list, pos_y_list, pos_z_list = [], [], []
    rot_x_list, rot_y_list, rot_z_list = [], [], []
    gripper_list = []

    for instance in input_jsonl:
        action = instance['conversations'][1]['raw_actions']
        action = [float(action[i]) for i in range(len(action))]
        pos_x_list.append(action[0])
        pos_y_list.append(action[1])
        pos_z_list.append(action[2])
        rot_x_list.append(action[3])
        rot_y_list.append(action[4])
        rot_z_list.append(action[5])
        gripper_list.append(action[6])

    total_list = [pos_x_list, pos_y_list, pos_z_list, rot_x_list, rot_y_list, rot_z_list, gripper_list]
    total_bin = []

    for index, individual_list in enumerate(total_list):
        action_series = pd.Series(individual_list)
        discretized_data, bins = pd.qcut(action_series, args.discretize_bins, labels=False, retbins=True, duplicates='drop')
        total_bin.append(bins)
        print(f"bin {index}:", bins)

    for index, instance in enumerate(input_jsonl):
        steps = []
        action = instance['conversations'][1]['raw_actions']
        input_jsonl[index]['raw_actions'] = action
        input_jsonl[index]['id'] = instance['id']
        input_jsonl[index]['image'] = instance['image']

        action = [float(action[i]) for i in range(len(action))]
        action_list = [assign_bin(action[i], total_bin[i]) for i in range(6)]
        action_list.append(int(action[6]))
        input_jsonl[index]['action'] = action_list 
        input_jsonl[index]['fields'] = '[instruction],[vision],action'
        instruction = instance['conversations'][0]['value'].replace('<image>\n', "")
        input_jsonl[index]['instruction'] = f"<s> You are a helpful assistant. USER: {instruction} ASSISTANT:"
        # delete the key 'conversations'
        del input_jsonl[index]['conversations']
        
        output_jsonl.append(input_jsonl[index])

    total_bin[-1] = [-0.5,0.5,1.5]

    # Save each category to its own JSONL file
    with open(args.output_filename, 'w') as outfile:
        for step in output_jsonl:
            outfile.write(json.dumps(step) + '\n')


    df = pd.DataFrame(total_bin)
    df.to_csv(args.csv_filename, index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON data and save processed actions and bins.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_filename', type=str, required=True, help='Path to save the output JSONL file prefix.')
    parser.add_argument('--csv_filename', type=str, required=True, help='Path to save the bins as a CSV file.')
    parser.add_argument('--discretize_bins', type=int, default=256, help='Number of bins to discretize the actions.')
    args = parser.parse_args()
    main(args)
