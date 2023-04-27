script_files = ['Mac1.py', 'Mac2.py', 'Mac3.py']

for script_file in script_files:
    with open(script_file, 'r') as file:
        script_content = file.read()
        exec(script_content)