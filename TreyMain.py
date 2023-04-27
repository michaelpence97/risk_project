script_files = ['Trey1.py', 'Trey2.py', 'Trey3.py']

for script_file in script_files:
    with open(script_file, 'r') as file:
        script_content = file.read()
        exec(script_content)