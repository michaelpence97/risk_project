script_files = ['Michael1.py', 'Michael2.py', 'Michael3.py']

for script_file in script_files:
    with open(script_file, 'r') as file:
        script_content = file.read()
        exec(script_content)
