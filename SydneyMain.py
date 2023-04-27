script_files = ['Sydney1.py', 'Sydney2.py', 'Sydney3.py']

for script_file in script_files:
    with open(script_file, 'r') as file:
        script_content = file.read()
        exec(script_content)