

import os

repo_path = r"C:\Users\asus\Desktop\formula1_predictions"  # Adjust this path if needed
output_file = 'all_python_scripts.txt'

with open(output_file, 'w', encoding='utf-8') as out_file:
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                out_file.write(f"{'='*80}\n{file_path}\n{'='*80}\n")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    out_file.write(content)
                    out_file.write('\n\n')

print(f"All Python scripts have been written to '{output_file}'.")
