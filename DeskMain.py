script_files = ['Desk1.py', 'Desk2.py', 'Desk3.py']

for script_file in script_files:
    with open(script_file, 'r') as file:
        script_content = file.read()
        exec(script_content)