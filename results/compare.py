import os
import csv
import re

def parse_metrics(file_path):
    """Parse a single metrics file and return the data as a dictionary."""
    metrics = {}
    current_metric = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^[\w-]+_(beat|downbeat)$', line):
                current_metric = line
                metrics[current_metric] = {}
            elif current_metric and ':' in line:
                dataset, value = line.split(':')
                metrics[current_metric][dataset.strip()] = float(value.strip())
    
    return metrics

def convert_to_csv(input_dir, output_csv):
    """Convert metrics files in a directory to a consolidated CSV file."""
    all_data = {}
    modules = []
    datasets = set()
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            module_name = os.path.splitext(file_name)[0]
            modules.append(module_name)
            file_path = os.path.join(input_dir, file_name)
            module_metrics = parse_metrics(file_path)
            
            for metric, data in module_metrics.items():
                if metric not in all_data:
                    all_data[metric] = {}
                all_data[metric][module_name] = data
                datasets.update(data.keys())

    datasets = sorted(datasets)

    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write header
        header = ['Metric', 'Dataset'] + modules
        writer.writerow(header)

        # Write data rows
        for metric, module_data in all_data.items():
            for dataset in datasets:
                row = [metric, dataset]
                for module in modules:
                    value = module_data.get(module, {}).get(dataset, '')
                    row.append(value)
                writer.writerow(row)


input_directory = "results/8_folds"  # 替换为您的目录路径
output_csv_file = "results/8_folds/metrics_comparison.csv"  # 输出的CSV文件名

convert_to_csv(input_directory, output_csv_file)
print(f"转换完成，结果已保存到 {output_csv_file}")