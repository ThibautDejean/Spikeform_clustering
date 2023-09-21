import os
import re

def get_valid_timer_names(labels_file_path):
    with open(labels_file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

def extract_timers_from_file(file_path, valid_timer_names):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    sections = re.split(r'NAME:\s*(\w+)', content)
    all_timers = []

    for i in range(1, len(sections), 2):
        timer_name = sections[i].strip()
        if timer_name in valid_timer_names:
            timers = re.findall(r'\+\d+\s*\+\d+\.\d+', sections[i+1])
            all_timers.extend([float(timer.split('+')[-1]) for timer in timers])

    return all_timers

def process_patient_directory(patient_directory, valid_timer_names, output_directory):
    all_timers = []

    for file in os.listdir(patient_directory):
        if file.endswith(".mrk"):
            file_path = os.path.join(patient_directory, file)
            timers = extract_timers_from_file(file_path, valid_timer_names)
            all_timers.extend(timers)

    output_file_path = os.path.join(output_directory, f"{os.path.basename(patient_directory)}.txt")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for timer in all_timers:
            output_file.write(str(timer) + '\n')

def process_main_directory(directory_path, labels_file_path, output_directory):
    valid_timer_names = get_valid_timer_names(labels_file_path)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for patient_name in os.listdir(directory_path):
        patient_directory = os.path.join(directory_path, patient_name)
        if os.path.isdir(patient_directory):
            process_patient_directory(patient_directory, valid_timer_names, output_directory)

main_directory_path = "/Users/thibautdejean/Desktop/IterativeLearningFeedback5"
labels_file_path = "/Users/thibautdejean/Desktop/All_labels.txt"
output_directory = "/Users/thibautdejean/Desktop/Timers2/"

process_main_directory(main_directory_path, labels_file_path, output_directory)
