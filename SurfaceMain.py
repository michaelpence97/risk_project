script_files = ['Surface1.py', 'Surface2.py', 'Surface3.py']

for script_file in script_files:
    with open(script_file, 'r') as file:
        script_content = file.read()
        exec(script_content)