from datetime import datetime
import pandas as pd
import numpy as np
import re
import os

log_path = './logs/'
logs = os.listdir(log_path)
results = []
init_dict = {'init0':'Random','init1':'Ontology','init3':'Model'}
number_snapshots = 5
primitive = 'Hits@3'
min_num = 1
max_num = 0

for log in logs:
    if '.log' in log:
        with open(log_path+log, "r") as file:
            log_data = file.read()
        
        params = log.replace('_','-').split('.log')[0].split('-')
        dataset = params[0]
        continual = params[2]
        init = init_dict[params[3]]
        max_epoch = int(params[4][:-1])
        if len(params)>5:
            rn = float(params[5][:-2])
        else:
            rn = 0

        metrics = [dataset,continual,init,max_epoch,rn]
        
        
        # Define refined regex patterns
        final_snapshot_block_pattern = r"Snapshot:(\d+).+?\+------------\+--------\+.*?\n((?:\|\s+\d+\s+\|.*?\n)+)"
        report_snapshot_row_pattern = r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        training_time_pattern = r"Sum_Training_Time:([\d.]+)"
        every_training_time_pattern = r"Every_Training_Time:\[([\d.,\s]+)\]"
        transfer_pattern = r"Forward transfer: ([\d.-]+)\s+Backward transfer: ([\d.-]+)"

        # Parse Final Result block
        metrics_log = []
        for block_match in re.finditer(final_snapshot_block_pattern, log_data, re.DOTALL):
            outer_snapshot = int(block_match.group(1))
            rows = block_match.group(2)
            
            for row_match in re.finditer(report_snapshot_row_pattern, rows):
                inner_snapshot, mrr, hits1, hits3, hits5, hits10 = row_match.groups()
                metrics_log.append({
                    "Outer Snapshot": outer_snapshot,
                    "Inner Snapshot": int(inner_snapshot),
                    "MRR": float(mrr),
                    "Hits@1": float(hits1),
                    "Hits@3": float(hits3),
                    "Hits@5": float(hits5),
                    "Hits@10": float(hits10)
                })

        metrics_log_df = pd.DataFrame(metrics_log).drop_duplicates()

        
        report_results_pattern = r"\|\s+(\d+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|"
        report_results_pattern = r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"

        report_results = []
        for row_match in re.finditer(report_results_pattern, log_data):
            snapshot, time, whole_mrr, whole_hits1, whole_hits3, whole_hits10 = row_match.groups()
            if float(time) > 0.5:
                report_results.append({
                    "Snapshot": int(snapshot),
                    "Time": float(time),
                    "Whole_MRR": float(whole_mrr),
                    "Whole_Hits@1": float(whole_hits1),
                    "Whole_Hits@3": float(whole_hits3),
                    "Whole_Hits@10": float(whole_hits10)
                })

        report_results_df = pd.DataFrame(report_results).drop_duplicates()
        metrics.append(report_results_df.iloc[4,2]) #MRR
        metrics.append(report_results_df.iloc[4,3]) #H@1
        metrics.append(report_results_df.iloc[4,4]) #H@3
        metrics.append(report_results_df.iloc[4,5]) #H@10

        ############### BASE ######################
        base = 0
        alpha_0_0 = metrics_log_df[(metrics_log_df['Outer Snapshot']==0)&(metrics_log_df['Inner Snapshot']==0)][primitive].values[0]
        for i in range(1,number_snapshots):

            alpha_i_0 = metrics_log_df[(metrics_log_df['Outer Snapshot']==i)&(metrics_log_df['Inner Snapshot']==0)][primitive].values[0]

            base += min(min_num,alpha_i_0/alpha_0_0)
        base /= (number_snapshots-1)
        metrics.append(np.round(base,3))
       

        ################ NEW ######################

        new = 0
        for i in range(1,number_snapshots):
            alpha_i_i = metrics_log_df[(metrics_log_df['Outer Snapshot']==i)&(metrics_log_df['Inner Snapshot']==i)][primitive].values[0]
            new += alpha_i_i
        new /= (number_snapshots-1)
        metrics.append(np.round(new,3))


        # Initialize variables
        snapshots = {}
        current_snapshot = None

        # Regular expression to extract timestamp, snapshot, and epoch information
        log_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO.*Snapshot:(\d+)\tEpoch:(\d+)")

        # Read and process the log file
        with open(log_path+log, 'r') as file:
            for line in file:
                match = log_pattern.search(line)
                if match:
                    timestamp_str = match.group(1)
                    snapshot_id = int(match.group(2))
                    epoch_id = int(match.group(3))

                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                    if snapshot_id not in snapshots:
                        snapshots[snapshot_id] = []
                    snapshots[snapshot_id].append(timestamp)

        # Compute time differences within each snapshot
        time_differences = []
        for snapshot_id, timestamps in snapshots.items():
            snapshot_differences = [
                (timestamps[i] - timestamps[i - 1]).total_seconds()
                for i in range(1, len(timestamps))
            ]
            time_differences.extend(snapshot_differences)

        # Compute the average time difference
        average_time_diff = sum(time_differences) / len(time_differences)
        metrics.append(average_time_diff)      
        results.append(metrics)

results = pd.DataFrame(results)
results.columns =['Dataset', 'Continual', 'Initialization', 'Max Epochs','Random Noise','MRR','Hits@1','Hits@3','Hits@10','Base','New','Time']

results.to_csv('results.csv',index=False)