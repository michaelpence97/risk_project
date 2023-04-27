script_files = ['Sam1.py', 'Sam2.py', 'Sam3.py']

for script_file in script_files:
    with open(script_file, 'r') as file:
        script_content = file.read()
        exec(script_content)