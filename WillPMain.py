script_files = ['WillP1.py', 'WillP2.py', 'WillP3.py']

for script_file in script_files:
    with open(script_file, 'r') as file:
        script_content = file.read()
        exec(script_content)
